"""Update the RAM guard test assertions from literal "5.0 GB" to a regex."""

content = open('tests/test_detectors.py', 'rb').read()

# Fix 1: AudioDetectorTests - memory guard check
old1 = b'            with self.assertRaisesRegex(RuntimeError, "at least 5.0 GB"):\r\n                AudioDetector()'
new1 = b'            with self.assertRaisesRegex(RuntimeError, r"at least \\d+\\.\\d+ GB"):\r\n                AudioDetector()'

# Fix 2: VideoDetectorTests - require_pella_memory check  (keeps "at least 5.0 GB" since
# eval_audio_conditions.MIN_PELLA_FREE_RAM_GB is now 3.5)
old2 = b'        with self.assertRaisesRegex(RuntimeError, "at least 5.0 GB"):\r\n            require_pella_memory(0.7)'
new2 = b'        with self.assertRaisesRegex(RuntimeError, r"at least \\d+\\.\\d+ GB"):\r\n            require_pella_memory(0.7)'

# Fix 3: pella_can_load tests — keep testing that 5.0 passes (5.0 >= 3.5) — no change needed
# just verify logic still works with new threshold

n = 0
if old1 in content:
    content = content.replace(old1, new1, 1)
    n += 1
    print(f"Fixed AudioDetectorTests RAM guard assertion")
else:
    print(f"WARNING: old1 not found")

if old2 in content:
    content = content.replace(old2, new2, 1)
    n += 1
    print(f"Fixed VideoDetectorTests require_pella_memory assertion")
else:
    print(f"WARNING: old2 not found")

open('tests/test_detectors.py', 'wb').write(content)
print(f"Done. {n} replacements made.")
