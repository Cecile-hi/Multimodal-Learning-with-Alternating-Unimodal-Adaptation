import os

test_audio_dir = '/data1/zhangxiaohui/IEMOCAP/subaudio/'

all_test_files = os.listdir('/data1/zhangxiaohui/IEMOCAP/subvideo')

for i, item in enumerate(all_test_files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(all_test_files)))
        print('*******************************************')
    mp4_filename = os.path.join('/data1/zhangxiaohui/IEMOCAP/subvideo/', item)
    wav_filename = os.path.join(test_audio_dir, item.replace(".mp4", ".wav"))
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))





