from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor

class Processor:
	def __init__(self):
		feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr = True, ocr_lang = 'eng')
		tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
		self.processor = LayoutLMv3Processor(feature_extractor, tokenizer)


	def __call__(self, img):
		encoding = self.processor(
			img,
			max_length = 512,
			padding = "max_length",
			truncation = True,
			return_tensors = "pt"
			)

		return encoding