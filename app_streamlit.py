import streamlit as st
import os
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import tempfile
import urllib.parse

st.title("📽 動画ハイライト自動生成アプリ（SNS共有付き・1本制限）")

uploaded_file = st.file_uploader("動画ファイルをアップロード (mp4)", type="mp4")

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ 動画ファイル {uploaded_file.name} をアップロードしました！")

        try:
            audio_path = os.path.join(tmpdir, "audio.wav")
            video = VideoFileClip(video_path)
            st.write("🔍 音声を抽出中...")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)

            st.write("🔍 音声特徴を抽出中...")
            y, sr = librosa.load(audio_path, sr=22050)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
            threshold = rms.mean() * 1.5
            highlight_times = times[rms > threshold]

            highlights = []
            start_time = None
            prev_time = None
            for t in highlight_times:
                if start_time is None:
                    start_time = t
                elif t - prev_time > 2.0:
                    highlights.append((start_time, prev_time))
                    start_time = t
                prev_time = t
            if start_time is not None:
                highlights.append((start_time, prev_time))

            filtered_highlights = []
            for start, end in highlights:
                duration = end - start
                if duration >= 10:
                    filtered_highlights.append((start, min(end, start + 20)))

            scored_highlights = []
            for start, end in filtered_highlights:
                start_idx = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
                end_idx = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
                score = rms[start_idx:end_idx].mean()
                scored_highlights.append((score, start, end))

            scored_highlights.sort(reverse=True)
            top_highlights = scored_highlights[:1]  # ★ 1本だけに制限

            st.info("🎬 ハイライト動画を生成中...")

            for i, (_, start, end) in enumerate(top_highlights):
                try:
                    clip = video.subclip(start, end)
                    output_path = os.path.join(tmpdir, f"highlight_{i+1}.mp4")
                    clip.write_videofile(output_path, codec="libx264", audio=True, audio_codec="aac", verbose=False, logger=None)

                    st.video(output_path)

                    # SNS共有用セクション
                    st.subheader("📤 SNS投稿内容（任意）")
                    user_text = st.text_area("投稿文やハッシュタグを入力（X向け）", height=100)
                    encoded_text = urllib.parse.quote(user_text)
                    if st.button("Xに投稿（リンクを開く）"):
                        x_url = f"https://twitter.com/intent/tweet?text={encoded_text}"
                        st.markdown(f"[→ Xで投稿する]({x_url})", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ ハイライト動画 #{i+1} の生成に失敗しました")
                    st.code(str(e), language="python")

            st.success("✅ ハイライト動画の生成が完了しました！")

        except Exception as e:
            st.error("❌ 処理全体でエラーが発生しました")
            st.code(str(e), language="python")
