import streamlit as st
from transcriber import Transcription
import docx
from datetime import datetime
import pathlib
import io
import json
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# app wide config
st.set_page_config(page_title="Whisper", layout="wide", page_icon="💬")

# load stylesheet
with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# app sidebar for uplading audio files
with st.sidebar.form("input_form"):
    input_files = st.file_uploader(
        "Files", type=["mp4", "m4a", "mp3", "wav"], accept_multiple_files=True
    )

    whisper_model = st.selectbox(
        "Whisper model", options=["tiny", "base", "small", "medium", "large"], index=4
    )

    pauses = st.checkbox("Pause detection", value=False)

    speaker_diarization = st.checkbox(
        "Speaker recognition (experimental)",
        value=False,
        help="Speaker recognition only works with clearly separated speaker segments and is not very reliable.",
    )

    transcribe = st.form_submit_button(label="Start")

if transcribe:
    if input_files:
        st.session_state.transcription = Transcription(input_files, speaker_diarization)
        st.session_state.transcription.transcribe(whisper_model)
    else:
        st.error("Please select a file")

# if there is a transcription, render it. If not, display instructions
if "transcription" in st.session_state:
    for i, output in enumerate(st.session_state.transcription.output):
        doc = docx.Document()
        save_dir = str(pathlib.Path(__file__).parent.absolute()) + "/transcripts/"
        st.markdown(f"#### Transcription from {output['name']}")
        st.markdown(
            f"_(whisper model:_`{whisper_model}` -  _language:_ `{output['language']}`)"
        )
        color_coding = st.checkbox(
            "Colour Coding",
            value=False,
            key={i},
            help="Color coding of a word based on the probability of it being correctly recognized. The color scale ranges from green (high) to red (low).",
        )
        prev_word_end = -1
        text = ""
        html_text = ""
        # Define the color map
        colors = [(0.6, 0, 0), (1, 0.7, 0), (0, 0.6, 0)]
        cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

        with st.expander("Transcript"):
            if speaker_diarization:
                speakers = {"SPEAKER_00": "A", "SPEAKER_01": "B"}
                for idx, group in enumerate(output["diarization"]):
                    try:
                        captions = json.load(
                            open(
                                f"{pathlib.Path(__file__).parent.absolute()}/buffer/{idx}.json"
                            )
                        )["segments"]
                    except Exception as ex:
                        print(ex)
                    if captions:
                        if idx == 0 and speakers.get(group[0].split()[-1], "") == "B":
                            speakers["SPEAKER_00"], speakers["SPEAKER_01"] = (
                                speakers["SPEAKER_01"],
                                speakers["SPEAKER_00"],
                            )
                        speaker = speakers.get(group[0].split()[-1], "")
                        if idx != 0:
                            html_text += "<br><br>"
                            text += "\n\n"
                        html_text += f"{speaker}: "
                        text += f"{speaker}: "
                        for c in captions:
                            for w in c["words"]:
                                if w["word"]:
                                    if (
                                        pauses
                                        and prev_word_end != -1
                                        and w["start"] - prev_word_end >= 3
                                    ):
                                        pause = w["start"] - prev_word_end
                                        pause_int = int(pause)
                                        html_text += (
                                            f'{"."*pause_int}{{{pause_int}sek}}'
                                        )
                                        text += f'{"."*pause_int}{{{pause_int}sek}}'
                                    prev_word_end = w["end"]
                                    if color_coding:
                                        rgba_color = cmap(w["probability"])
                                        rgb_color = tuple(
                                            round(x * 255) for x in rgba_color[:3]
                                        )
                                    else:
                                        rgb_color = [0, 0, 0]
                                    html_text += f"<span style='color:rgb{rgb_color}'>{w['word']}</span>"
                                    text += w["word"]
            else:
                for idx, segment in enumerate(output["segments"]):
                    for w in output["segments"][idx]["words"]:
                        # check for pauses in speech longer than 3s
                        if (
                            pauses
                            and prev_word_end != -1
                            and w["start"] - prev_word_end >= 3
                        ):
                            pause = w["start"] - prev_word_end
                            pause_int = int(pause)
                            html_text += f'{"."*pause_int}{{{pause_int}sek}}'
                            text += f'{"."*pause_int}{{{pause_int}sek}}'
                        prev_word_end = w["end"]
                        if color_coding:
                            rgba_color = cmap(w["probability"])
                            rgb_color = tuple(round(x * 255) for x in rgba_color[:3])
                            print(w["word"], w["probability"], rgb_color)
                        else:
                            rgb_color = [0, 0, 0]
                        html_text += (
                            f"<span style='color:rgb{rgb_color}'>{w['word']}</span>"
                        )
                        text += w["word"]
                        # insert line break if there is a punctuation mark
                        if any(c in w["word"] for c in "!?.") and not any(
                            c.isdigit() for c in w["word"]
                        ):
                            html_text += "<br><br>"
                            text += "\n\n"
            st.markdown(html_text, unsafe_allow_html=True)
            doc.add_paragraph(text)

        # save transcript as docx. in local folder
        file_name = (
            output["name"]
            + "-"
            + whisper_model
            + "-"
            + datetime.today().strftime("%d-%m-%y")
            + ".docx"
        )
        doc.save(save_dir + file_name)

        bio = io.BytesIO()
        doc.save(bio)
        st.download_button(
            label="Download Transcript",
            data=bio.getvalue(),
            file_name=file_name,
            mime="docx",
        )

else:
    # show instruction page
    st.markdown(
        """<h1>Patsy's Transcriber</h1> 
            <p>Credits: The master's thesis by <a href='mailto:johanna.jaeger89@icloud.com'>Johanna Jäger<a/> 
            using <a href='https://openai.com/blog/whisper'>OpenAI Whisper</a>.</p> 
            <h2 class='highlight'>DATA PRIVACY:</h2> 
            <p>The program runs locally. 
                Transcripts are saved to a local directory on this PC.</p>
            <h2 class='highlight'>USAGE:</h2> 
            <ol>
                <li>Select the files you want to transcribe (multiple files possible)</li>
                <li>Select a model (<i>large</i> for the best result) and other parameters, then click 'Start'</li>
                <li>View the resulting transcripts in the <i>transcripts</i> folder of this directory</li>
            </ol>
        """,
        unsafe_allow_html=True,
    )
