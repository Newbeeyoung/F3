import os
from PIL import Image
from torch.utils.data import Dataset
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images


class caption_dataset_llava(Dataset):
    def __init__(self, image_dir, tokenizer, image_processor, model_config, conv_mode, limit=1000):
        self.image_dir = image_dir
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.limit = limit

        # Collect image files
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(exts)]
        files.sort()
        self.image_files = files[: self.limit]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_id = os.path.splitext(image_name)[0]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        # Build captioning prompt
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + "Describe the image in one sentence."
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + "Describe the image in one sentence."

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return image_id, image_tensor, input_ids, image.size


