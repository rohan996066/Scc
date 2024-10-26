import logging
from io import BytesIO
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext
from diffusers import StableDiffusionPipeline
import torch

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

# Define the start command
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! Send me a text prompt and I will generate an image for you.')

# Define the message handler
def generate_image(update: Update, context: CallbackContext) -> None:
    prompt = update.message.text
    logger.info(f"Received prompt: {prompt}")
    
    # Generate the image
    with torch.no_grad():
        image = pipe(prompt).images[0]
    
    # Convert the image to a BytesIO  object
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Send the image back to the user
    update.message.reply_photo(photo=img_byte_arr)

# Define the error handler
def error(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Update {update} caused error {context.error}')

def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater("7368453739:AAGdc8TBzbxv5q15jQKDqH6jxJMqhBlpkak")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register the start command handler
    dispatcher.add_handler(CommandHandler("start", start))

    # Register the message handler
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, generate_image))

    # Register the error handler
    dispatcher.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()