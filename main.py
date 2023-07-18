import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from htmlTemplates import css
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
from pydub import AudioSegment
import io


def get_caption(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    raw_image = Image.open(image).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    preds = tokenizer.batch_decode(out, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
    # return out

    # model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    # feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    # tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    #
    # max_length = 16
    # num_beams = 4
    # gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    #
    # images = []
    #
    # i_image = Image.open(image)
    # if i_image.mode != "RGB":
    #     i_image = i_image.convert(mode="RGB")
    #
    # images.append(i_image)
    #
    # pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    # pixel_values = pixel_values.to(device)
    #
    # output_ids = model.generate(pixel_values, **gen_kwargs)
    #
    # preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # preds = [pred.strip() for pred in preds]
    # return preds


def main():
    st.set_page_config(page_title="Image Caption Generator", page_icon=":camera_with_flash:")

    st.write(css, unsafe_allow_html=True)
    st.header("Image Caption Generator :camera_with_flash:")
    st.write("Upload an image and click on 'generate a caption'")

    # File uploader
    image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if st.button("Generate a caption 	:loud_sound:"):
        with st.spinner("Generating..."):
            # Function call to caption generator model
            generated_caption = get_caption(image)
            st.write("Generated Caption of the uploaded image:")

            caption = generated_caption.pop(0)
            st.write(caption)
            # processings for audio
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            inputs = processor(text=caption, return_tensors="pt")
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

            audio = sf.write("speech.wav", speech.numpy(), samplerate=16000)
            audio_data = AudioSegment.from_file("speech.wav")

            # Convert audio data to a byte stream
            audio_stream = io.BytesIO()
            audio_data.export(audio_stream, format='wav')
            audio_stream.seek(0)

            # Display the audio playback in Streamlit
            st.audio(audio_stream, format='audio/wav')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
