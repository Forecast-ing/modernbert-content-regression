from transformers import pipeline

classifier = pipeline("text-classification", model="Forecast-ing/modernBERT-content-regression")

import gradio as gr


emails = [
    "Revolutionize Your NLP with modernBERT!\n\nUnleash the full potential of your natural language processing tasks with modernBERT – the cutting-edge language model that’s faster, more accurate, and optimized for real-world applications. Whether you're building chatbots, search engines, or sentiment analysis tools, modernBERT is here to elevate your projects to new heights. Try it today and experience the difference!",
    
    "Modern NLP Starts Here: modernBERT\n\nAre you ready to supercharge your NLP pipelines? Meet modernBERT – the next generation of language understanding. It’s lightweight, developer-friendly, and built for scalability. From startups to enterprises, modernBERT is transforming how teams handle language data. Get started now and see how it can simplify and amplify your workflows.",
    
    "Faster. Smarter. modernBERT.\n\nWhy settle for outdated models when you can have modernBERT? Designed with speed and accuracy in mind, modernBERT delivers unparalleled performance for tasks like text classification, entity recognition, and more. Plus, it integrates seamlessly into your existing stack. Upgrade to modernBERT and stay ahead of the curve!",
    
    "The NLP Solution You've Been Waiting For\n\nDiscover modernBERT: the all-in-one NLP model built for the modern era. With state-of-the-art performance, minimal latency, and maximum versatility, modernBERT is redefining what’s possible in natural language processing. Don’t wait – take your NLP projects to the next level with modernBERT today!",
    
    "Transform Your Data Insights with modernBERT\n\nModern problems require modern solutions. That’s why we created modernBERT – a language model tailored to meet the demands of today’s fast-paced data-driven world. Its powerful architecture ensures high accuracy and blazing-fast predictions. See how modernBERT can revolutionize your business – try it for free now!"
]



def predict(text):
    prediction = classifier(text)
    return f"{prediction[0]['score'] / 100:.2%}"

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label="Written Content", lines=6)],
    outputs=[gr.Textbox(label="Predicted Clickthrough Rate (industry standard is around 2%)", lines=1)],
    examples= emails,
    flagging_mode="never",
)

demo.launch()