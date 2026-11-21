"""Standalone window media and score-bound pagination, without cloud calls."""

import base64
import json
import math
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidates

SEED = "11111111-1111-4111-8111-111111111111"


def candidate(candidate_id='one', score=0.1, offset=30, created=None):
    return SimpleNamespace(
        candidate_id=candidate_id, score=score, priority=score + 0.1, window_id=f'window-{candidate_id}',
        query_species_id='DENEBR', recording_id='recording',
        created_at=created or datetime(2026, 1, 1, tzinfo=timezone.utc),
        audio_gcs_uri=f'gs://osa-frog-data/windows/v1/window-{candidate_id}.wav',
        spectrogram_gcs_uri=None, start_seconds=offset, end_seconds=offset + 5,
    )


class MediaUrls:
    expires_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def url(self, uri):
        return 'signed:' + uri if uri else None


class CandidateTests(unittest.TestCase):
    def test_standalone_audio_plays_zero_to_five_with_original_offsets_as_metadata(self):
        row = candidate(offset=725)
        payload = candidates._candidate_payload(row, MediaUrls())
        self.assertEqual((payload['start_seconds'], payload['end_seconds']), (0, 5))
        self.assertEqual((payload['recording_start_seconds'], payload['recording_end_seconds']),
                         (725, 730))
        self.assertEqual(payload['audio_url'], 'signed:' + row.audio_gcs_uri)
        self.assertIsNone(payload['spectrogram_url'])

    def test_invalid_database_windows_are_rejected_instead_of_serving_wrong_audio_bounds(self):
        for start, end in [(-1, 4), (0, 4), (0, 6), (float('nan'), 5),
                           (0, float('inf')), (True, 6)]:
            row = candidate()
            row.start_seconds, row.end_seconds = start, end
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                candidates._candidate_payload(row, MediaUrls())
        for uri in [None, '', 'https://example.com/window.wav']:
            row = candidate()
            row.audio_gcs_uri = uri
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                candidates._candidate_payload(row, MediaUrls())

    def test_priority_keyset_handles_ties_and_cutoff_without_skips_or_joins(self):
        cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
        # SQL priorities are deliberately different from score order.
        rows = [candidate('a', 0), candidate('b', 1), candidate('c', 0.2), candidate('d', 0.5),
                candidate('new', 0, created=cutoff + timedelta(seconds=1))]
        for row, priority in zip(rows, [3, 0.1, 0.1, 2, 0.01]):
            row.priority = priority
        queries = []

        def read_rows(query, parameters):
            queries.append(query)
            params = {p.name: p.value for p in parameters}
            self.assertEqual(params['seed'], SEED)
            eligible = [row for row in rows if row.created_at <= params['cutoff']]
            if params['after_priority'] is not None:
                boundary = (params['after_priority'], params['after_id'])
                eligible = [row for row in eligible if (row.priority, row.candidate_id) > boundary]
            return sorted(eligible, key=lambda row: (row.priority, row.candidate_id))[:params['fetch_count']]

        with patch.object(candidates.settings, 'read_rows', side_effect=read_rows), \
                patch.object(candidates.media, 'MediaUrls', MediaUrls):
            first = candidates.get_candidates('DENEBR', 1, cutoff, None, None, SEED)
            self.assertEqual(first['candidates'][0]['candidate_id'], 'b')
            second = candidates.get_candidates('DENEBR', 2, *candidates.decode_cursor(first['next_cursor'], 'DENEBR'))
            self.assertEqual([row['candidate_id'] for row in second['candidates']], ['c', 'd'])
            final = candidates.get_candidates('DENEBR', 2, *candidates.decode_cursor(second['next_cursor'], 'DENEBR'))
            self.assertEqual([row['candidate_id'] for row in final['candidates']], ['a'])
            self.assertIsNone(final['next_cursor'])
            replay = candidates.get_candidates('DENEBR', 1, cutoff, None, None, SEED)
            self.assertEqual(replay, first)

        for query in queries:
            self.assertIn('ORDER BY priority ASC, candidate_id ASC', query)
            self.assertIn('@after_priority IS NULL', query)
            self.assertIn('priority = @after_priority AND candidate_id > @after_id', query)
            self.assertIn('c.created_at <= @cutoff', query)
            self.assertIn('c.audio_gcs_uri', query)
            self.assertNotIn('JOIN', query)
            self.assertNotIn('.annotations`', query)
            self.assertNotIn('RAND()', query)

    def test_old_malformed_or_changed_order_cursors_require_restart(self):
        cursor = {'v': 3, 'species_id': 'DENEBR', 'cutoff': '2026-01-02T00:00:00+00:00',
                  'priority': 0.1, 'candidate_id': 'one', 'seed': SEED, 'temperature': 0.5}
        changes = [{'v': 1}, {'v': 2}, {'temperature': 1}, {'temperature': True},
                   {'seed': ''}, {'seed': 'bad'}, {'species_id': 'AGACAL'}]
        changes += [{'priority': value} for value in [float('nan'), float('inf'), float('-inf'), 0, -1, None, True, '0.1']]
        for fields in changes:
            token = base64.urlsafe_b64encode(json.dumps({**cursor, **fields}).encode()).decode()
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, 'restart'):
                candidates.decode_cursor(token, 'DENEBR')
        for priority in [1e-15, 0.25, 1, 100.12345678912345]:
            row = candidate()
            row.priority = priority
            token = candidates.encode_cursor('DENEBR', datetime(2026, 1, 2, tzinfo=timezone.utc), row, SEED)
            self.assertEqual(candidates.decode_cursor(token, 'DENEBR')[1:], (priority, 'one', SEED))

    def test_uniform_bins_avoid_signed_overflow_and_float_endpoints(self):
        for fingerprint in [-(2**63), -(2**63)+1, -1, 0, 1, 2**63-1]:
            uniform = (float(fingerprint & (2**52-1)) + 0.5) / 2**52
            self.assertGreater(uniform, 0)
            self.assertLess(uniform, 1)
            for score in [0, 0.5, 1]:
                priority = -math.log(uniform) * math.exp(score / 0.5)
                self.assertTrue(math.isfinite(priority))
                self.assertGreater(priority, 0)

    def test_exact_id_refresh_reads_one_table_and_returns_only_remaining_ids(self):
        rows = [candidate('one')]
        with patch.object(candidates.settings, 'read_rows', return_value=rows) as read, \
                patch.object(candidates.media, 'MediaUrls', MediaUrls):
            data = candidates.refresh_candidate_media('DENEBR', ['one', 'removed'])
        self.assertEqual([item['candidate_id'] for item in data['candidates']], ['one'])
        self.assertIsNone(data['next_cursor'])
        query, parameters = read.call_args.args
        self.assertNotIn('JOIN', query)
        self.assertIn('.candidates`', query)
        self.assertEqual(parameters[1].values, ['one', 'removed'])


if __name__ == '__main__':
    unittest.main()
