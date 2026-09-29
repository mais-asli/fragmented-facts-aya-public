"""Verify and unpack the actual Aya research outputs sent by the TAU workflow."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', default='results/tau_reviewed_screen_workflow.json')
    parser.add_argument('--destination', default='results/tau_research_20260926')
    parser.add_argument('--final-status', default='behavior_analysis_finished')
    parser.add_argument('--archive', help='Verified local archive streamed by the Aya held-out workflow')
    args = parser.parse_args()
    workflow = (ROOT / args.workflow).resolve()
    destination = (ROOT / args.destination).resolve()
    if not workflow.is_relative_to((ROOT / 'results').resolve()) or not destination.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Workflow and destination must be inside project results')
    state = json.loads(workflow.read_text(encoding='utf-8'))
    if state.get('status') != 'workflow_finished' or state.get('workflow_exit_code') != 0:
        raise ValueError('The TAU Aya research workflow has not finished successfully')
    final = state['workflow']
    if final.get('status') != args.final_status:
        raise ValueError('The final remote event is not a completed Aya analysis')
    if args.archive:
        archive_path = (ROOT / args.archive).resolve()
        if not archive_path.is_relative_to((ROOT / 'results').resolve()):
            raise ValueError('Archive must be inside project results')
        payload = archive_path.read_bytes()
        expected_size = state['bundle_bytes']
        expected_hash = state['bundle_sha256']
        final_size = final.get('bundle_bytes', final.get('archive_bytes'))
        final_hash = final.get('bundle_sha256', final.get('archive_sha256'))
        if (expected_size != final_size or expected_hash != final_hash):
            raise ValueError('Local and remote bundle claims differ')
    else:
        payload = base64.b64decode(final['research_bundle_base64'], validate=True)
        expected_size = final['research_bundle_bytes']
        expected_hash = final['research_bundle_sha256']
    if (len(payload) != expected_size or
            hashlib.sha256(payload).hexdigest() != expected_hash):
        raise ValueError('Research bundle checksum mismatch')
    if destination.exists():
        raise FileExistsError('Research outputs already extracted; preserve the first copy')
    destination.mkdir(parents=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts:
                raise ValueError('Unsafe archive path')
            target = destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Archive path escapes output directory')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError('Archive member has no contents')
                with target.open('xb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                count += 1
            else:
                raise ValueError('Research bundle contains a link or special file')
    print(json.dumps({'status': 'verified_and_extracted',
                      'sha256': expected_hash,
                      'files': count, 'directory': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
