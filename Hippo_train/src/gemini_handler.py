# src/gemini_handler.py
import google.generativeai as genai
import logging
import time
import os # For os.getenv

from src import config # To access GEMINI_API_KEY and LOG_LEVEL

# Ensure logger is configured based on config.LOG_LEVEL
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiHandler:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.model = None
        self.image_model = None # Placeholder if you use a specific vision/multimodal model later

        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in config or .env file. GeminiHandler will not function.")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            # Choose a model. Check https://ai.google.dev/models/gemini for latest/suitable ones.
            # 'gemini-1.5-flash-latest' is good for speed and cost-effectiveness.
            # 'gemini-pro' was a common one, 'gemini-1.0-pro' if being specific.
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            
            # If you need image understanding capabilities (not generation from this API directly)
            # self.image_model = genai.GenerativeModel('gemini-pro-vision')
            
            logger.info("Gemini API configured successfully with model 'gemini-1.5-flash-latest'.")
        except Exception as e:
            logger.error(f"Error configuring Gemini API or initializing model: {e}")
            self.model = None # Ensure model is None if setup fails

    def _generate_with_retry(self, prompt_parts, max_retries=3, delay_seconds=5, generation_config=None):
        if self.model is None:
            logger.error("Gemini model not initialized. Cannot generate content.")
            return "Error: Gemini model not initialized."
        
        # Default generation config (can be overridden)
        # See https://ai.google.dev/docs/ πί_reference/python/google.generativeai.types.GenerationConfig
        default_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=['...'],
            # max_output_tokens=2048, # Default depends on model, e.g. gemini-1.0-pro is 2048
            temperature=0.7, # Controls randomness. Lower for more factual, higher for creative.
            # top_p=1.0,
            # top_k=1
        )
        current_gen_config = generation_config if generation_config else default_config

        for attempt in range(max_retries):
            try:
                # The generate_content method can take various types for prompt_parts:
                # str, list of str, list of Part objects (for multimodal)
                response = self.model.generate_content(prompt_parts, generation_config=current_gen_config)
                
                # Accessing response text safely
                if hasattr(response, 'text') and response.text:
                    return response.text
                elif response.parts: # Handle cases where response might be in parts
                    full_text = "".join([part.text for part in response.parts if hasattr(part, 'text')])
                    if full_text:
                        return full_text
                
                logger.warning(f"Gemini response (attempt {attempt + 1}) did not contain text or expected structure.")
                logger.debug(f"Full Gemini response object: {response}")
                # Check for safety ratings or finish reason if no text
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.error(f"Content generation blocked. Reason: {response.prompt_feedback.block_reason_message}")
                    return f"Error: Content generation blocked ({response.prompt_feedback.block_reason_message})"
                # Check candidates for finish reason
                if response.candidates:
                    for cand in response.candidates:
                        if cand.finish_reason != 1: # 1 is "STOP" (successful)
                             logger.warning(f"Candidate finished with reason: {cand.finish_reason.name}. Message: {cand.finish_message}")


                # If still no text, return error for this attempt before retry
                if attempt < max_retries - 1:
                    logger.info(f"Retrying Gemini request in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                else: # Last attempt failed to produce text
                    return "Error: Received no parsable text from Gemini after multiple attempts."

            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                if "API key not valid" in str(e) or "permission" in str(e).lower():
                    logger.critical("Gemini API key seems invalid or lacks permissions. Please check.")
                    return "Error: Gemini API key invalid or permission issue."
                if " Billing account not found" in str(e): # Specific check
                    logger.critical("Gemini API error: Billing account not found or not associated with the project.")
                    return "Error: Gemini API billing account issue."

                if attempt < max_retries - 1:
                    time.sleep(delay_seconds * (attempt + 1)) # Exponential backoff might be better
                else:
                    return f"Error: Gemini API failed after {max_retries} retries due to: {e}"
        
        return "Error: Gemini API generation failed after all retries." # Should be caught by loop end

    def generate_zookeeper_alert(self, hippo_id, behavior, duration, potential_implication, advice):
        prompt = f"""
        Generate a concise zookeeper alert based on the following information for a hippo:
        Hippo ID: {hippo_id}
        Observed Behavior: {behavior}
        Duration of Behavior: {duration}
        Potential Implication of this behavior: {potential_implication}
        Suggested Action/Observation: {advice}

        The alert should be professional, direct, and actionable for zookeeping staff.
        Focus on clarity and brevity.
        Example: "ALERT: Hippo {hippo_id} observed {behavior} for {duration}. This may indicate {potential_implication}. Suggestion: {advice}."
        """
        return self._generate_with_retry(prompt)

    def generate_daily_digest(self, hippo_id, activities_summary_str, persona_name="Moodeng"):
        """
        Args:
            hippo_id (str): Identifier for the hippo.
            activities_summary_str (str): A string summarizing activities, e.g.,
                                      "morning: swimming for 1 hour, feeding for 30 mins; afternoon: resting for 2 hours, short walk."
            persona_name (str): Name to use for the hippo persona.
        """
        prompt = f"""
        You are {persona_name}, a playful and curious pygmy hippo. Your ID is {hippo_id}.
        Today, your activities were: {activities_summary_str}.

        Write a short, fun diary entry about your day from your perspective as {persona_name}.
        Make it engaging for children and families who might read it on a zoo's website.
        Include some hippo-like thoughts or feelings (e.g., about food, water, naps, friends).
        Keep the diary entry under 150 words.
        Example style: "Hi everyone, it's {persona_name}! Today was so much fun! I splashed in my pool all morning – whee! Then I had some yummy greens. After that, a long nap in the warm mud felt just right. Can't wait for tomorrow!"
        """
        return self._generate_with_retry(prompt)

    def generate_educational_snippet(self, observed_behavior, hippo_id="A hippo", context_details=""):
        prompt = f"""
        Subject: Educational snippet about hippo behavior.
        Hippo Identifier: {hippo_id}
        Observed Behavior: {observed_behavior}
        Additional Context (if any): {context_details}

        Task: Provide a concise (around 100-150 words) and engaging educational explanation for the general public about why hippos (specifically pygmy hippos, if contextually appropriate) exhibit the observed behavior.
        Cover these points:
        1. A brief, clear explanation of the behavior.
        2. Why hippos typically exhibit this behavior (e.g., physiological need, social signal, environmental interaction).
        3. The significance or purpose of this behavior.
        4. Include one interesting or fun fact related to this behavior or hippo biology/ecology.

        The tone should be informative, accessible, and fascinating.
        Example for "wallowing in mud":
        "{hippo_id.capitalize()} was seen wallowing in the mud! Hippos do this to protect their sensitive skin from the sun and biting insects, as they don't have much hair. The mud acts like a natural sunscreen and bug repellent! It also helps keep them cool. Fun fact: Hippo skin secretes an oily red substance sometimes called 'blood sweat', which also offers sun protection and might have antibiotic properties!"
        """
        return self._generate_with_retry(prompt)

    def generate_image_concept_prompt(self, behavior, mood="natural", environment="typical hippo habitat", style="photorealistic"):
        prompt = f"""
        Create a detailed and vivid text prompt suitable for an AI image generator (like Midjourney, DALL-E, or Stable Diffusion).
        The image should depict a pygmy hippo.

        Key elements to include in the generated prompt:
        - Subject: Pygmy hippo.
        - Featured Behavior: {behavior}
        - Pose/Action: Describe the hippo's pose clearly related to the behavior.
        - Expression (if applicable for the behavior and mood): e.g., contented, alert, yawning.
        - Environment Details: {environment} (e.g., in murky water with lily pads, on a muddy bank, amidst dense green foliage, in a zoo enclosure with specific features).
        - Lighting: (e.g., soft morning light, dappled sunlight through canopy, dramatic evening light, artificial enclosure lighting).
        - Artistic Style & Mood: {style} with a {mood} feeling (e.g., photorealistic and serene, whimsical illustration and playful, detailed macro shot and curious, National Geographic style and wild).
        - Camera View/Composition: (e.g., close-up, wide shot, eye-level view).
        - Negative Prompts (optional, if you want to suggest avoiding certain things): e.g., "avoid cartoony, avoid humans".

        Generate a compelling and descriptive prompt.
        Example for behavior "yawning": "Ultra-realistic close-up, eye-level view of a pygmy hippo mid-yawn, showcasing its pink tongue and small tusks. The hippo is partially submerged in murky, still water reflecting the overcast sky. Background shows dense, out-of-focus jungle foliage. Shot with a telephoto lens, creating a shallow depth of field. Mood: natural, slightly imposing. Style: National Geographic wildlife photography."

        Generate a new prompt for the behavior '{behavior}', mood '{mood}', environment '{environment}', style '{style}':
        """
        return self._generate_with_retry(prompt)

    def generate_voice_narration_script(self, text_content, persona="friendly_educator", target_audience="general_public"):
        prompt = f"""
        Adapt the following text content to be a natural-sounding script for voice narration.
        Persona for Narration: {persona} (e.g., "friendly_educator", "calm_documentarian", "playful_hippo_character", "enthusiastic_child_narrator").
        Target Audience: {target_audience} (e.g., "general_public", "children_6_to_10", "zoo_visitors").

        Consider:
        - Breaking long sentences into shorter ones.
        - Adding natural pauses (you can indicate with ellipses ... or new lines).
        - Suggesting emphasis on certain words (e.g., by using ALL CAPS for a word or a phrase like *really important*).
        - Making the tone conversational and engaging, appropriate for the persona and audience.
        - If the text is already well-suited, you can return it with minor tweaks or confirm it's good as is.

        Original Text:
        ---
        {text_content}
        ---

        Adapted Narration Script (Return only the script itself):
        """
        return self._generate_with_retry(prompt)


if __name__ == '__main__':
    # Ensure .env file exists and GEMINI_API_KEY is set
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", ".env")): # Check in parent dir
        with open(os.path.join(os.path.dirname(__file__), "..", ".env"), "w") as f:
             f.write("GEMINI_API_KEY=YOUR_API_KEY_HERE_IN_ROOT_DOT_ENV\n")
        logger.warning("Created a dummy .env file in project root. Please fill it with your Gemini API key.")

    # Reload config in case .env was just created (though getenv should be dynamic)
    # from importlib import reload
    # reload(config) # This can be problematic depending on how config is structured.
    # A better approach for __main__ is to instantiate GeminiHandler which reads config.

    handler = GeminiHandler()

    if handler.model: # Check if model initialized successfully
        logger.info("--- Testing Zookeeper Alert ---")
        alert = handler.generate_zookeeper_alert(
            hippo_id="Hippo02 (Pygmy)",
            behavior="lethargic and not eating for 6 hours",
            duration="6 hours",
            potential_implication="illness or severe stress",
            advice="Veterinary check immediately, observe for other symptoms, ensure access to fresh water and preferred food items away from other hippos if stressed."
        )
        print(f"Zookeeper Alert:\n{alert}\n")

        logger.info("--- Testing Daily Digest ---")
        digest = handler.generate_daily_digest(
            hippo_id="H001-MoodengJr",
            activities_summary_str="Morning: lots of swimming and making big splashes! Afternoon: long nap in the comfy mud, then munching on yummy greens and a bit of fruit. Evening: quietly watched the birds.",
            persona_name="Muddy"
        )
        print(f"Daily Digest for Muddy:\n{digest}\n")

        logger.info("--- Testing Educational Snippet ---")
        edu_behavior = "a mother hippo nudging her calf towards the water"
        snippet = handler.generate_educational_snippet(
            observed_behavior=edu_behavior,
            hippo_id="A mother pygmy hippo and her calf",
            context_details="The calf appeared hesitant."
        )
        print(f"Educational Snippet for '{edu_behavior}':\n{snippet}\n")

        logger.info("--- Testing Image Concept Prompt ---")
        img_prompt = handler.generate_image_concept_prompt(
            behavior="two young pygmy hippos playfully wrestling in shallow water",
            mood="joyful and energetic",
            environment="a sunlit clearing in a dense rainforest, with muddy banks and lush green ferns",
            style="hyperrealistic digital painting, capturing water splashes and textures"
        )
        print(f"Image Concept Prompt:\n{img_prompt}\n")

        logger.info("--- Testing Voice Narration Script (using previous snippet) ---")
        if "Error:" not in snippet: # Only proceed if snippet generation was successful
            narration_script = handler.generate_voice_narration_script(
                text_content=snippet,
                persona="warm and knowledgeable nature documentary narrator",
                target_audience="families and nature enthusiasts"
            )
            print(f"Voice Narration Script:\n{narration_script}\n")
        else:
            logger.warning("Skipping voice narration test as educational snippet generation failed.")
    else:
        logger.error("Gemini Handler could not be initialized. Check API key (in .env file in project root) and network connectivity.")
        logger.error("Ensure your Gemini API key has the necessary permissions and billing is set up if required by Google AI Studio/Vertex AI.")