import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

import src.api_server as api_server


class AudioPreviewSuccessCleanupContractTests(unittest.TestCase):
    def test_success_returns_mp3_and_background_task_removes_output(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original_named_tempfile = tempfile.NamedTemporaryFile
                captured = {}
                background_tasks = api_server.BackgroundTasks()

                def create_temp_file(**kwargs):
                    return original_named_tempfile(dir=directory, **kwargs)

                def fake_ffmpeg(command, **kwargs):
                    captured["command"] = command
                    Path(command[-1]).write_bytes(b"generated mp3 preview")
                    return type(
                        "CompletedProcess",
                        (),
                        {"returncode": 0, "stderr": b""},
                    )()

                upload = UploadFile(
                    filename="recording.flac",
                    file=io.BytesIO(b"original audio"),
                )

                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=create_temp_file,
                    ),
                    patch.object(
                        api_server.subprocess,
                        "run",
                        side_effect=fake_ffmpeg,
                    ),
                ):
                    response = await api_server.convert_audio_preview(
                        background_tasks,
                        upload,
                    )

                command = captured["command"]
                self.assertIn("-vn", command)
                self.assertEqual(command[command.index("-ac") + 1], "2")
                self.assertEqual(command[command.index("-ar") + 1], "44100")
                self.assertEqual(command[command.index("-b:a") + 1], "128k")
                self.assertEqual(response.media_type, "audio/mpeg")
                self.assertEqual(response.filename, "recording.mp3")
                self.assertTrue(Path(response.path).is_file())
                self.assertTrue(upload.file.closed)
                self.assertEqual(
                    [path.name for path in Path(directory).iterdir()],
                    [Path(response.path).name],
                )

                await background_tasks()
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
