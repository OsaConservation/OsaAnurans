"""Standalone window media and score-bound pagination, without cloud calls."""

import base64
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidates


def candidate(candidate_id='one', score=0.1, offset=30, created=None):
    return SimpleNamespace(
        candidate_id=candidate_id, score=score, window_id=f'window-{candidate_id}',
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

    def test_score_keyset_handles_zero_ties_and_creation_cutoff_without_recording_join(self):
        cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
        rows = [candidate('b', 0), candidate('a', 0), candidate('d', 1),
                candidate('c', 0.2), candidate('new', 0, created=cutoff + timedelta(seconds=1))]
        queries = []

        def read_rows(query, parameters):
            queries.append(query)
            params = {parameter.name: parameter.value for parameter in parameters}
            eligible = [row for row in rows if row.created_at <= params['cutoff']]
            if params['after_score'] is not None:
                boundary = (params['after_score'], params['after_id'])
                eligible = [row for row in eligible if (row.score, row.candidate_id) > boundary]
            return sorted(eligible, key=lambda row: (row.score, row.candidate_id))[:params['fetch_count']]

        with patch.object(candidates.settings, 'read_rows', side_effect=read_rows), \
                patch.object(candidates.media, 'MediaUrls', MediaUrls):
            first = candidates.get_candidates('DENEBR', 1, cutoff, None, None)
            self.assertEqual(first['candidates'][0]['candidate_id'], 'a')
            first_cursor = first['next_cursor']
            after_cutoff, score, after_id = candidates.decode_cursor(first_cursor, 'DENEBR')
            self.assertEqual((score, after_id), (0, 'a'))
            second = candidates.get_candidates('DENEBR', 2, after_cutoff, score, after_id)
            self.assertEqual([row['candidate_id'] for row in second['candidates']], ['b', 'c'])
            final = candidates.get_candidates(
                'DENEBR', 2, *candidates.decode_cursor(second['next_cursor'], 'DENEBR')
            )
            self.assertEqual([row['candidate_id'] for row in final['candidates']], ['d'])
            self.assertIsNone(final['next_cursor'])
            replay = candidates.get_candidates('DENEBR', 1, cutoff, None, None)
            self.assertEqual(replay['next_cursor'], first_cursor)

        for query in queries:
            self.assertIn('ORDER BY c.score ASC, c.candidate_id ASC', query)
            self.assertIn('@after_score IS NULL', query)
            self.assertIn('c.score = @after_score AND c.candidate_id > @after_id', query)
            self.assertIn('c.created_at <= @cutoff', query)
            self.assertIn('c.audio_gcs_uri', query)
            self.assertNotIn('.recordings`', query)
            self.assertNotIn('c.created_at >', query)

    def test_v1_cursor_requires_restart_and_v2_score_must_be_numeric_finite_in_range(self):
        cursor = {'v': 2, 'species_id': 'DENEBR', 'cutoff': '2026-01-02T00:00:00+00:00',
                  'score': 0.1, 'candidate_id': 'one'}
        for patch_fields in [{'v': 1}, {'score': float('nan')}, {'score': float('inf')},
                             {'score': float('-inf')}, {'score': -0.1}, {'score': 1.1},
                             {'score': None}, {'score': True}, {'score': '0.1'},
                             {'species_id': 'AGACAL'}]:
            token = base64.urlsafe_b64encode(json.dumps({**cursor, **patch_fields}).encode()).decode()
            with self.subTest(fields=patch_fields), self.assertRaisesRegex(ValueError, 'restart'):
                candidates.decode_cursor(token, 'DENEBR')

        for value in [0, 0.25, 1]:
            token = candidates.encode_cursor('DENEBR', datetime(2026, 1, 2, tzinfo=timezone.utc),
                                             candidate(score=value))
            self.assertEqual(candidates.decode_cursor(token, 'DENEBR')[1], value)


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
