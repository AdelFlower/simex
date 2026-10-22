"""Build a thin research image from the project's pinned CLI, without auth files."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def main():
    package = ROOT / '.tools/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl'
    cli = package / 'bin/codex'
    version = subprocess.check_output([str(cli), '--version'], text=True).strip()
    if version != 'codex-cli 0.162.1':
        raise RuntimeError('Install the project .tools lockfile; CLI must be 0.162.1')
    with tempfile.TemporaryDirectory(prefix='simex-image-') as temporary:
        directory = Path(temporary)
        for file in (ROOT / 'docker/research').iterdir():
            if file.is_file():
                shutil.copy2(file, directory / file.name)
        for source, name in ((cli, 'codex'), (package / 'bin/codex-code-mode-host', 'codex-code-mode-host'),
                             (package / 'codex-path/rg', 'rg'),
                             (Path(imageio_ffmpeg.get_ffmpeg_exe()), 'ffmpeg')):
            shutil.copy2(source, directory / name)
        subprocess.run(['docker', 'build', '-t', 'simex-research:0.162.1', str(directory)], check=True)


if __name__ == '__main__':
    main()
