from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
import beam_search
import top_sampling

app = Flask(__name__)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-recipe-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-recipe-generation")
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator, tokenizer

# Load the model globally for efficiency (consider caching for larger models)
generator, tokenizer = load_model()

def generate_recipe(ingredients):
    all_ingredients = ", ".join(ingredients)
    
    # Generate recipe using chosen logic
    if sampling_mode == "Beam Search":
        generated = generator(all_ingredients, return_tensors=True, return_text=False, **beam_search.generate_kwargs)
        outputs = beam_search.post_generator(generated, tokenizer)
    elif sampling_mode == "Top-k Sampling":
        generated = generator(all_ingredients, return_tensors=True, return_text=False, **top_sampling.generate_kwargs)
        outputs = top_sampling.post_generator(generated, tokenizer)
    output = outputs[0]
    return output

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe_api():
    data = request.get_json()
    ingredients = data["ingredients"]
    recipe = generate_recipe(ingredients)
    return jsonify(recipe)

if __name__ == '__main__':
    sampling_mode = "Beam Search"  # or "Top-k Sampling"
    app.run(debug=True)
