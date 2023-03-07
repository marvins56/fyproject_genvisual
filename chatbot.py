# import os
import pyttsx3  # converts text to speech
import openai

# get API key from top-right dropdown on OpenAI website
openai.api_key = "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"


engine = pyttsx3.init('sapi5')
# sapi5 is an API and the technology for voice recognition and synthesis provided by Microsoft
voices = engine.getProperty('voices')  # gets you the details of the current voices
engine.setProperty('voice', voices[1].id)  # 0-male voice , 1-female voice




def speak(audio):  # function for assistant to speak
    engine.say(audio)
    engine.runAndWait()  # without this command, the assistant won't be audible to us

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return speak(res['choices'][0]['text'].strip())


query = (
    "how to create a luganda chatbot"
)

complete(query)



# //The best training method to use for fine-tuning a pre-trained model with
# sentence transformers is the Masked Language Model (MLM) training. MLM training involves
# randomly masking some of the words in a sentence and then training the model to predict the masked words. This helps the model to learn the context
# of th