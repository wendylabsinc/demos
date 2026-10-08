"""Operator action boundary: a page load never releases robot ownership."""
import json
import threading
import tempfile
from pathlib import Path
import unittest
import urllib.request
import urllib.error
from unittest.mock import Mock,patch
from http.server import ThreadingHTTPServer
import operator_panel as panel

class OperatorPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.deploying=Path(self.directory.name)/'deployment.lock'
        self.marker=patch.object(panel,'DEPLOYING',self.deploying);self.marker.start()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),panel.Handler)
        self.server.wendy=Mock();self.server.wendy.request.return_value={'phase':'PASSIVE','adapter_initialized':False}
        self.port=patch.object(panel,'PORT',self.server.server_port);self.port.start()
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2);self.port.stop()
        self.marker.stop();self.directory.cleanup()
    def post(self,data,token=panel.TOKEN,path='/release'):
        request=urllib.request.Request(self.url+path,data=json.dumps(data).encode(),
            headers={'Content-Type':'application/json','X-Operator-Token':token,'Origin':self.url})
        return urllib.request.urlopen(request,timeout=2)
    def test_page_load_only_exposes_enabled_operator_button(self):
        with urllib.request.urlopen(self.url,timeout=2) as response:html=response.read().decode()
        self.assertIn('<button id="release">',html)
        self.assertIn('/frame.jpg?t=',html)
        self.assertNotIn('iframe',html)
        self.assertIn('<button id="start"',html)
        self.assertIn('<button id="stop"',html)
        self.server.wendy.request.assert_not_called()
    def test_missing_support_or_wrong_local_token_cannot_release(self):
        for data,token in [({'can_supported':False},panel.TOKEN),({'can_supported':True},'bad')]:
            with self.subTest(data=data,token=token),self.assertRaises(urllib.error.HTTPError):self.post(data,token)
        self.server.wendy.request.assert_not_called()
    def test_explicit_supported_release_calls_backend_once(self):
        with self.post({'can_supported':True}) as response:self.assertEqual(json.load(response)['phase'],'PASSIVE')
        self.server.wendy.request.assert_called_once_with(release=True)

    def test_start_and_stop_are_explicit_separate_operator_requests(self):
        with self.post({},path='/start'):pass
        self.server.wendy.request.assert_called_once_with(action='start')
        self.server.wendy.request.reset_mock()
        with self.post({},path='/stop'):pass
        self.server.wendy.request.assert_called_once_with(action='stop')

    def test_start_is_blocked_during_deployment_but_stop_remains_available(self):
        self.deploying.write_text('updating')
        with self.assertRaises(urllib.error.HTTPError) as failure:self.post({},path='/start')
        self.assertEqual(failure.exception.code,409)
        self.assertIn('update in progress',json.load(failure.exception)['error'])
        self.server.wendy.request.assert_not_called()
        with self.post({},path='/stop'):pass
        self.server.wendy.request.assert_called_once_with(action='stop')

    def test_backend_start_failure_reaches_page_and_is_not_reported_as_success(self):
        self.server.wendy.request.side_effect=RuntimeError('Controller unavailable')
        with self.assertRaises(urllib.error.HTTPError) as failure:self.post({},path='/start')
        self.assertEqual(json.load(failure.exception)['error'],'Controller unavailable')
        self.server.wendy.request.assert_called_once_with(action='start')
        self.assertIn('id="action-status"',panel.PAGE)
        self.assertIn('Start request failed:',panel.PAGE)
        self.assertIn('Start accepted',panel.PAGE)

if __name__=='__main__':unittest.main()
