![20231212060523_image](https://github.com/SimonHanlyJones/AI-boleth/assets/46434944/8faf66c3-5822-4932-ac6a-a8cfe46fd772)# AI-boleth
Intel Advent of Gen AI winning submission

Simon Hanly-Jones, Emmanuel Isaac, Ryan Mann

This project was created for the December 2023 Intel Advent of Gen Ai Hackathon, which required submissions the be in a Jupyter Notebook. The winning submission has been slightly modified to work in a Google Colab notebook with a 16GB GPU, rather than relying on the proprietary API's provided during the competition.

It operates as an enhanced monster manual for Dungeons and Dragons 5e monsters, giving users accurate, stylised information about their abilities, an image of the monster as well as a short video.

Intel's blogpost about this project can be found here, together with a writeup of the other winners:

https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Advent-of-GenAI-Hackathon-Recap-of-the-Final-Challenge-Custom/post/1556584

## Here a summary of the workflow

Step 1 -> User asks a question about a DnD monster

Step 2 -> Zephyr bot with RAG context injection answers the question like a cheesy villain

Step 3 -> the subject of the question, the monster is identified from the RAG record

Step 4 -> the monster is then fed into another zephyr chatbot which generates a good image prompt for SDXL

Step 5 -> the prompt is used to generate an image (sdxl)

Step 6 -> the image is used to generate a video (sdv-xt)

We have made a GUI so that the user can jump on the pipeline at any point numbered above from 1 to 4.
They can ask the question about the monster, and get the answer, enhanced prompt, image and video.
They can just out a creature into the prompt enhancer to get an image and video
Then can put in their original prompt with no enhancement and get their own image and video.
This means that we have built arbitrary text to video generation.

The the bot answers, prompt enhancements and output images perform very well. Far beyond our expectations. The videos are a bit hit and miss.
Everything is contained in the notebook and it can be run with the included GUI. The demo_sequence() function can also be uncommented and run to generate about 15 examples of the full pipeline. It takes about 35 mins to run.

EXAMPLES

Please have a look at our examples. We have been honest about cherry picking the best performing videos. The folders contain a full output of our demo_sequence(), and some cherry picked good and bad videos.

REQUIREMENTS

All of our testing was done on Google Colab V100 GPU's which are 16GB of vram.

Interesting Technical Stuff

## PIPELINES STEP 2: ZEPHYR BOT WITH RAG AND CONVERSATION

Getting the chat prompt right for this first bot was very challenging. After much trial and error we discovered a prompt that worked:

        _monster_prompt_template = _monster_prompt_template = (
            "<|system|>You are a helpful assistant who is extremely knowledgeable about Dungeons and Dragons 5th Edition monsters. Your job is to give a comprehensive answer to the question from the provided context. Do so in the manner of an overacted, but honest villain.</s>\n"
            "<|user|>\n"
            "Context: {conversational_context}{info_context}\n"
            "Question: {query}\n"
            "</s>\n"
            "<|assistant|>\n"
        )
        
Interestingly, "context" could not be split into headings like "conversation history" and "monster information". The model just couldn't follow the structure. Wizard code performed much better with headings like this in the previous challenge, which is something to keep in mind for future projects.
We looked at the source code and found a template, as well as the template on Predication Guard (which was extremely helpful), but there was no formal guidance on formatting these prompts.
We found it necessary to include both sets of information because without the chat history the bot thought it was a monster itself. Without the informational context from RAG it hallucinated.

It is also worth noting that special formatting for past messages made performance worse. The best performance was obtained using a list without removing the python artefact, eg:

        self.messages = [
            {
                "user": "What is the armor class of an Aboleth?\n",
                "assistant": "The slippery Aboleth! Its feeble armor class is a mere 17, a laughable attempt at self-preservation through natural armor.\n"
            },
            {
                "user": "Is the Aboleth evil?\n",
                "assistant": "Evil, you ask? Chaotic evil courses through the veins of the wretched Aboleth! A malevolent force, craving nothing but chaos and destruction.\n"
            }
        ]

Sometimes the bot will exaggerate the performance of a monster. For instance it will correctly identify a monster is weak in a certain aspect, but will then say it is in fact a strength for example it generated this answer about a Kobold, which is a creature of low inelegance

### QUESTION:
```
How smart is a Kobold?
```
### ANSWER:
```
My dear foes, do not underestimate the cunning of the vile Kobold! With its low Intelligence score of 8 and a Intelligence modifier of -1, this small humanoid may seem like a simple creature, but its wits are sharper than you might think! Its Pack Tactics trait allows it to work in unison with its allies, making it a formidable force to be reckoned with! And let us not forget its Sunlight Sensitivity, which forces it to adapt and strategize in order to overcome the disadvantage it faces in sunlight. Beware its approach, for it will strike with the ferocity of a thunderbolt, its cunning mind guiding it every step of the way!
```
It will also append commentary in square brackets like this:
```
[A Kobold is in fact of limited intellect, but an evil villain would not admit that]
```
We started stopping responses early by stopping at the first newline. We mention it here because it is interesting behavior. We have worked on this prompt a lot and it is possible that it is no happening anymore.

### STEP 4 PROMPT AUGMENTATION WITH ZEPHYR

As noted above, we use a second chatbot to augment prompts for the the image generation model and the video generation model. When reviewing the early video output we saw that images where the subject was in action worked better, we speculate the model can identify common movement patterns and can apply them better when it is more obvious what should be happening. For instance, a dragon in flight animated much better than a dragon sitting still.

This creates an issue for generic prompting. Put rhetorically, how do we know what action is appropriate for a creature when do don't know what the creature is. We initial tried prompting for something like:
```
a dramatic image of a {monster} in its normal environment
```
This was okay, but we saw a dramatic improvement with some more work. We refined our prompt to:
```
Concept art of a {monster} with a dramatic background, perfect anatomy, artstation, concept art, high definition, accent lighting
```
However, it did not consistently create the motion that we wanted.

We used a second Zephyr bot with the following instructions:

        messages = [
            {
                "role": "system",
                "content": "You are an AI language model that provides text-based responses to user inputs. Your responses should be relevant, accurate, and concise, and should reflect the capabilities and limitations of your programming. You should also strive to provide helpful and informative responses that add value to the user's experience."
            },
            {
                "role": "user",
                "content": f'improve this prompt for stable diffusion XL 1.0 base, incorporate an action or movement that would be appropriate for the given creature: "Concept art of a {monster} with a dramatic background, perfect anatomy, artstation, concept art, high definition, accent lighting" Respond only with improved prompts.'
            }]
            
This was a very standard system instruction obtained by asking mistral instruct what it used by default.
We saw greatly improved results to the video output with the augmented prompts. They were also quire poetic for example:
```
"Sea hag gliding through the depths of the ocean, her long, sinuous body undulating with each stroke. Dramatic underwater ruins loom in the background, casting eerie shadows on her scales. Perfect anatomy and intricate details bring this mythical creature to life, while accent lighting highlights her menacing features. High definition and artstation quality ensure a stunning final product."

"Epic scene of a majestic Pegasus soaring through a vibrant sunset sky, showcasing intricate anatomy and dynamic motion, crafted with precision and detail for ArtStation-worthy concept art in stunning high definition with captivating accent lighting."
```
The images for these prompts are included in the folder for the full test case.

![20231212060523_image](https://github.com/SimonHanlyJones/AI-boleth/assets/46434944/6a5dfb63-0396-441a-b802-46a13b28c3a9)

![20231212060549_image](https://github.com/SimonHanlyJones/AI-boleth/assets/46434944/a052a369-2583-4d02-8865-cea52230f0de)

### STEP 5 TEXT TO VIDEO MODEL SELECTION

We has some difficulty was our text to image model. We initially used sdxl-turbo. It is a fast checkpoint of sdxl and runs in low vram environments. However, it did not perform well when given the rectangular resolution of 576x1024 required by the image to video model. It generated a lot of additional limbs.
As such we switched to sdxl which performs very well.

### STEP 6 VIDEO GENERATION

As noted above, the videos can be inconsistent. The model likely performs better with real things that move in a predictable way. There is a input in our GUI which allows the user to put their own prompt in to investigate further.

## Running the project

The project can be run in the provided notebook. A 16gb GPU will be required, so a GPU enabled colab environment is ideal. The API's provided by Prediction Guard for the competition have been disabled, so we have modified the code so everything runs in the notebook environment.



 
