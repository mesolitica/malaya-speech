import json
import curses, time


def input_char(message):
    try:
        win = curses.initscr()
        win.addstr(0, 0, message)
        while True:
            ch = win.getch()
            if ch in range(32, 127):
                break
            time.sleep(0.05)
    except:
        raise
    finally:
        curses.endwin()
    return chr(ch)


with open('shuffled-iium.json') as fopen:
    texts = json.load(fopen)

try:
    with open('rejected-iium.json') as fopen:
        rejected = json.load(fopen)

except:
    rejected = {}

import queue
import sys
import sounddevice as sd
import soundfile as sf
import os

os.system('mkdir audio-iium')
device_info = sd.query_devices('LCS USB Audio')
device = None
samplerate = int(44100)
channels = 1
subtype = 'PCM_24'

for no, text in enumerate(texts):
    if str(no) in rejected:
        continue
    try:
        filename = f'audio-iium/{no}.wav'
        if os.path.isfile(filename):
            continue

        c = input_char(
            f'say: {text} , press `c` to continue or press any key except `c` to skip.'
        )
        print(c)
        if c.lower() == 'q':
            break
        if c.lower() != 'c':
            rejected[str(no)] = True
            continue

        q = queue.Queue()

        def callback(indata, frames, time, status):
            if status:
                print(status, file = sys.stderr)
            q.put(indata.copy())

        with sf.SoundFile(
            filename,
            mode = 'x',
            samplerate = samplerate,
            channels = channels,
            subtype = subtype,
        ) as file:
            with sd.InputStream(
                samplerate = samplerate,
                device = device,
                channels = channels,
                callback = callback,
            ):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                time.sleep(0.1)
                print('say: %s' % (text))
                print('#' * 80, '\n')

                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print(
            '\nRecording finished: %s, left %d\n'
            % (repr(filename), len(texts) - no)
        )

    except Exception as e:
        print(e)
        pass


with open('rejected-iium.json', 'w') as fopen:
    json.dump(rejected, fopen)
