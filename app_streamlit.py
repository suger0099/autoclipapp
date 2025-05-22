import streamlit as st
import os
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import tempfile

st.title("📽 単一ハイライト抽出アプリ（Render最適化版）")

uploaded_file = st.file_uploader("動画ファイルをアップロード (mp4)", type="mp4")

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ 動画ファイルをアップロードしました")

        try:
            audio_path = os.path.join(tmpdir, "audio.wav")
            video = VideoFileClip(video_path)
            st.write("🔍 音声抽出中...")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)

            st.write("🔍 音声特徴を解析中...")
            y, sr = librosa.load(audio_path, sr=22050)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            threshold = rms.mean() * 1.5
            loud_indices = np.where(rms > threshold)[0]

            if len(loud_indices) == 0:
                st.warning("📭 音量の高い部分が見つかりませんでした")
            else:
                start_time = times[loud_indices[0]]
                end_time = times[loud_indices[-1]]
                duration = min(end_time - start_time, 10.0)

                st.info(f"🎬 ハイライト: {start_time:.2f}s ～ {start_time+duration:.2f}s")
                clip = video.subclip(start_time, start_time + duration)
                output_path = os.path.join(tmpdir, "highlight.mp4")
                clip.write_videofile(output_path, codec="libx264", audio=True, audio_codec="aac", verbose=False, logger=None)
                st.video(output_path)

        except Exception as e:
            st.error("❌ エラーが発生しました")
            st.code(str(e), language="python")
