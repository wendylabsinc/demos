import unittest
from preview import connection_failure


class ConnectionFailureTests(unittest.TestCase):
    def test_cli_warning_and_json_become_separate_diagnostic(self):
        result = connection_failure('WARN TLS issuer hints paused\n'
            '{"error":{"code":"device_unreachable","message":"Connection refused", "retryable":true}}')
        self.assertEqual(result['code'], 'device_unreachable')
        self.assertEqual(result['diagnostic'], 'Connection refused')
        self.assertNotIn('WARN', result['detail'])
        self.assertNotIn('{', result['detail'])

    def test_identity_failure_is_preserved(self):
        result = connection_failure('{"error":{"code":"device_identity_mismatch","message":"Saved pin differs"}}')
        self.assertEqual(result['code'], 'device_identity_mismatch')

    def test_timeout_has_no_command_or_frame(self):
        result = connection_failure(timed_out=True)
        self.assertEqual(result['code'], 'station_timeout')
        self.assertEqual(result['diagnostic'], 'The authenticated device request timed out.')

    def test_unstructured_output_is_bounded_and_only_in_details(self):
        result = connection_failure('x' * 9000)
        self.assertEqual(len(result['diagnostic']), 4000)
        self.assertEqual(result['detail'], 'The station is temporarily unavailable.')


class ForwardTests(unittest.TestCase):
    def test_reuses_live_forward_instead_of_spawning_for_each_request(self):
        from unittest.mock import Mock, patch
        from preview import WendyForward
        forward = WendyForward('worker')
        forward.process = Mock()
        forward.process.poll.return_value = None
        forward.opened = True
        forward.port = 12345
        with patch('preview.subprocess.Popen') as spawn:
            self.assertEqual(forward.ensure(), 12345)
            self.assertEqual(forward.ensure(), 12345)
            spawn.assert_not_called()

    def test_interrupted_post_is_not_replayed(self):
        from unittest.mock import Mock, patch
        from urllib.error import URLError
        from preview import WendyForward
        forward = WendyForward('worker')
        forward.ensure = Mock(return_value=12345)
        forward.close = Mock()
        opener = Mock()
        opener.open.side_effect = URLError('connection closed')
        with patch('preview.urllib.request.build_opener', return_value=opener):
            with self.assertRaises(ConnectionError):
                forward.request('POST', '/api/events', b'{"event_id":"one-frame"}')
        self.assertEqual(opener.open.call_count, 1)
        forward.close.assert_called_once()
