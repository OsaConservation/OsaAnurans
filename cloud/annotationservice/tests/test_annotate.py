from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotate
from helpers import annotation_payload


class AnnotationTests(unittest.TestCase):
    def test_valid_payload_preserves_window_and_species(self):
        data = annotation_payload()
        data['annotator_name'] = '  Maria  '
        result = annotate.validate_submission(data)
        self.assertEqual(result['annotator_name'], 'Maria')
        self.assertEqual(result['annotations'], data['annotations'])

    def test_candidate_id_is_optional(self):
        data = annotation_payload()
        del data['annotations'][0]['candidate_id']
        result = annotate.validate_submission(data)
        self.assertIsNone(result['annotations'][0]['candidate_id'])

    def test_duplicates_and_unknown_relationships_are_accepted(self):
        data = annotation_payload()
        data['annotations'][0].update(window_id='not-in-database', query_species_id='UNKNOWN')
        data['annotations'] *= 2
        self.assertEqual(len(annotate.validate_submission(data)['annotations']), 2)

    def test_basic_format_checks_remain(self):
        changes = [dict(annotator_name=' '), dict(annotator_id='invalid'),
                   dict(annotations=[]), dict(annotations=[{}]),
                   dict(annotations=[dict(candidate_id='one', response='present')])]
        for change in changes:
            data = annotation_payload()
            data.update(change)
            with self.subTest(change=change), self.assertRaises(annotate.SubmissionError):
                annotate.validate_submission(data)
        for field, value in [('window_id', ''), ('query_species_id', None),
                             ('candidate_id', 'a\x00b'), ('response', 'yes')]:
            data = annotation_payload()
            data['annotations'][0][field] = value
            with self.subTest(field=field), self.assertRaises(annotate.SubmissionError):
                annotate.validate_submission(data)

    def test_twenty_answers_allowed_but_twenty_one_rejected(self):
        data = annotation_payload()
        data['annotations'] *= 20
        self.assertEqual(len(annotate.validate_submission(data)['annotations']), 20)
        data['annotations'].append(deepcopy(data['annotations'][0]))
        with self.assertRaises(annotate.SubmissionError):
            annotate.validate_submission(data)

    def test_one_parameterized_insert_and_no_existing_job_lookup(self):
        client = Mock()
        job = client.query.return_value
        job.num_dml_affected_rows = 1
        data = annotation_payload()
        result = annotate.submit_annotations(data, client, 'osaanurans', 'osafrogdataset',
                                              'us-south1', 100_000_000)
        self.assertEqual(result, {'submission_id': data['submission_id'], 'accepted_count': 1})
        client.query.assert_called_once()
        client.get_job.assert_not_called()
        call = client.query.call_args
        query = call.args[0]
        self.assertEqual(query.count('INSERT INTO'), 1)
        self.assertIn('FROM UNNEST(JSON_QUERY_ARRAY(@answers))', query)
        self.assertNotIn('JOIN', query)
        self.assertNotIn('ASSERT', query)
        self.assertNotIn('BEGIN TRANSACTION', query)
        self.assertNotIn('Maria', query)
        self.assertIn('GENERATE_UUID()', query)
        self.assertNotIn('job_id', call.kwargs)
        self.assertIsNone(call.kwargs['job_retry'])
        config = call.kwargs['job_config']
        params = {p.name: p.value for p in config.query_parameters}
        self.assertEqual(params['annotator_name'], 'Maria')
        self.assertEqual(config.maximum_bytes_billed, 100_000_000)

    def test_repeated_submissions_make_separate_inserts(self):
        data, client = annotation_payload(), Mock()
        client.query.return_value.num_dml_affected_rows = 1
        for _ in range(2):
            annotate.submit_annotations(data, client, 'osaanurans', 'osafrogdataset',
                                         'us-south1', 100_000_000)
        self.assertEqual(client.query.call_count, 2)
        client.get_job.assert_not_called()

    def test_write_timeout_propagates_without_automatic_resubmission(self):
        client = Mock()
        client.query.return_value.result.side_effect = TimeoutError('pending')
        with self.assertRaises(TimeoutError):
            annotate.submit_annotations(annotation_payload(), client, 'osaanurans',
                                         'osafrogdataset', 'us-south1', 100_000_000)
        client.query.assert_called_once()


if __name__ == '__main__':
    unittest.main()
