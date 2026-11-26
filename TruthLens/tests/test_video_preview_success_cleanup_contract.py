import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, UploadFile

import src.api_server as api_server


class VideoPreviewSuccessCleanupContractTests(unittest.TestCase):
    def test_success_returns_mp4_and_defers_output_cleanup_until_response_finishes(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as directory:
                original_named_tempfile = tempfile.NamedTemporaryFile
                captured = {}
                background_tasks = BackgroundTasks()

                def create_temp_file(**kwargs):
                    return original_named_tempfile(dir=directory, **kwargs)

                def fake_ffmpeg(command, **kwargs):
                    captured["command"] = command
                    Path(command[-1]).write_bytes(b"generated browser-compatible mp4")
                    return type(
                        "CompletedProcess",
                        (),
                        {"returncode": 0, "stderr": b""},
                    )()

                upload = UploadFile(
                    filename="sample.video.mp4",
                    file=io.BytesIO(b"original video bytes"),
                )
                with (
                    patch.object(api_server.shutil, "which", return_value="ffmpeg"),
                    patch.object(
                        api_server.tempfile,
                        "NamedTemporaryFile",
                        side_effect=create_temp_file,
                    ),
                    patch.object(api_server.subprocess, "run", side_effect=fake_ffmpeg),
                ):
                    response = await api_server.convert_video_preview(
                        background_tasks,
                        upload,
                    )

                command = captured["command"]
                self.assertEqual(command[command.index("-c:v") + 1], "libx264")
                self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
                self.assertEqual(command[command.index("-movflags") + 1], "+faststart")
                self.assertEqual(command[command.index("-c:a") + 1], "aac")
                self.assertEqual(command[command.index("-b:a") + 1], "128k")
                self.assertEqual(response.media_type, "video/mp4")
                self.assertEqual(
                    response.headers["content-disposition"],
                    'attachment; filename="sample.video_preview.mp4"',
                )
                self.assertTrue(Path(response.path).is_file())
                self.assertEqual(
                    [path.name for path in Path(directory).iterdir()],
                    [Path(response.path).name],
                )
                self.assertTrue(upload.file.closed)

                await background_tasks()
                self.assertEqual(list(Path(directory).iterdir()), [])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
