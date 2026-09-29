"""Focused safety and resume checks for the SSH upload receiver."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tau_upload
from tau_upload_receiver import receive_file, run, safe_path


class UploadChecks(unittest.TestCase):
    def setUp(self):
        test_base = Path(tempfile.gettempdir()) / 'tau-upload-checks'
        test_base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_base)
        self.root = Path(self.temp.name)
        if not self.root.resolve().is_relative_to(test_base.resolve()):
            raise ValueError('Unexpected test cleanup target')
        self.addCleanup(self.temp.cleanup)
        self.body = b'actual-file-bytes\x00\xff' * 10000
        self.item = {'path': 'models/shard.bin', 'size': len(self.body),
                     'sha256': hashlib.sha256(self.body).hexdigest()}

    def feed(self, body, item=None):
        item = item or self.item
        incoming = io.BytesIO(json.dumps({'action': 'continue', 'path': item['path']}).encode() + b'\n' + body)
        outgoing = io.BytesIO()
        receive_file(self.root, item, incoming, outgoing)
        return [json.loads(line) for line in outgoing.getvalue().splitlines()]

    def test_full_transfer_publishes_verified_file(self):
        events = self.feed(self.body)
        self.assertEqual((self.root / self.item['path']).read_bytes(), self.body)
        self.assertEqual(events[-1]['event'], 'file_verified')

    def test_resume_checks_prefix_and_appends_remaining_bytes(self):
        partial = self.root / '.upload/models/shard.bin.part'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(self.body[:12345])
        events = self.feed(self.body[12345:])
        self.assertEqual(events[0]['offset'], 12345)
        self.assertEqual(events[0]['prefix_sha256'], hashlib.sha256(self.body[:12345]).hexdigest())
        self.assertEqual((self.root / self.item['path']).read_bytes(), self.body)

    def test_bad_hash_never_publishes(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.feed(b'X' * len(self.body))
        self.assertFalse((self.root / self.item['path']).exists())
        self.assertTrue((self.root / '.upload/models/shard.bin.part').exists())

    def test_interruption_keeps_resumable_prefix(self):
        with self.assertRaises(EOFError):
            self.feed(self.body[:135])
        self.assertEqual((self.root / '.upload/models/shard.bin.part').read_bytes(), self.body[:135])

    def test_existing_different_file_is_preserved(self):
        target = self.root / self.item['path']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user-edited-data')
        with self.assertRaisesRegex(ValueError, 'left unchanged'):
            self.feed(self.body)
        self.assertEqual(target.read_bytes(), b'user-edited-data')

    def test_verified_file_is_skipped_without_reading_input(self):
        self.feed(self.body)
        output = io.BytesIO()
        receive_file(self.root, self.item, io.BytesIO(), output)
        self.assertEqual(json.loads(output.getvalue())['event'], 'file_already_verified')

    def test_known_setup_version_can_be_updated(self):
        item = dict(self.item, path='project_plan/feasibility/setup.py',
                    replace_sha256=hashlib.sha256(b'previous-script').hexdigest())
        target = self.root / item['path']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'previous-script')
        self.feed(self.body, item)
        self.assertEqual(target.read_bytes(), self.body)

    def test_unrecognized_setup_edit_is_preserved(self):
        item = dict(self.item, path='project_plan/feasibility/setup.py',
                    replace_sha256=hashlib.sha256(b'previous-script').hexdigest())
        target = self.root / item['path']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'independent-user-edit')
        with self.assertRaisesRegex(ValueError, 'left unchanged'):
            self.feed(self.body, item)
        self.assertEqual(target.read_bytes(), b'independent-user-edit')

    def test_paths_cannot_escape(self):
        for relative in ('../outside', '/outside', 'a/../../outside', 'C:/outside', 'a\\outside'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_path(self.root, relative)

    def test_symlink_is_not_followed(self):
        target = self.root / 'real'
        target.mkdir()
        try:
            (self.root / 'linked').symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest('OS does not permit symlink creation for this test')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            safe_path(self.root, 'linked/file')

    def test_complete_protocol_writes_receipt_without_inference(self):
        manifest = {'model_id': 'test-fixture', 'revision': 'test-fixture', 'files': [self.item]}
        incoming = io.BytesIO(json.dumps(manifest).encode() + b'\n' +
                             json.dumps({'action': 'continue', 'path': self.item['path']}).encode() +
                             b'\n' + self.body)
        output = io.BytesIO()
        run(self.root, incoming, output)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]['event'], 'complete')
        receipt = json.loads((self.root / 'results/tau-upload-receipt.json').read_text())
        self.assertFalse(receipt['inference_run'])
        self.assertFalse(receipt['slurm_job_submitted'])

    def test_sender_and_receiver_complete_over_real_binary_pipes(self):
        local = self.root / 'local'
        remote = self.root / 'remote'
        (local / 'scripts').mkdir(parents=True)
        receiver = Path(__file__).with_name('tau_upload_receiver.py').read_text()
        (local / 'scripts/tau_upload_receiver.py').write_text(receiver)
        source = local / 'source.bin'
        source.write_bytes(self.body)
        manifest = {'model_id': 'fixture', 'revision': 'fixture', 'files': [self.item],
                    'transfer_scope': 'code_only'}
        real_popen = subprocess.Popen
        receiver_command = ('from pathlib import Path; import sys; '
                            'from tau_upload_receiver import run; '
                            'run(Path(sys.argv[1]), sys.stdin.buffer, sys.stdout.buffer)')
        def local_receiver(command, **kwargs):
            self.assertIn('StrictHostKeyChecking=yes', command)
            return real_popen([sys.executable, '-c', receiver_command, str(remote)],
                              cwd=Path(__file__).parent, **kwargs)
        with patch.object(tau_upload, 'ROOT', local), patch.object(tau_upload, 'REPORT', local / 'unused.json'), \
             patch.object(tau_upload, 'prepare', return_value=(manifest, {self.item['path']: source})), \
             patch.object(tau_upload.subprocess, 'Popen', side_effect=local_receiver), \
             patch.object(sys, 'argv', ['tau_upload.py', '--code-only']), \
             patch.dict(os.environ, {'WINDIR': 'unused-test-path'}):
            tau_upload.main()
        self.assertEqual((remote / self.item['path']).read_bytes(), self.body)
        report = json.loads((local / 'results/tau_workflow.json').read_text())
        self.assertEqual(report['status'], 'uploaded_and_sha256_verified')
        self.assertFalse(report['inference_run'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
