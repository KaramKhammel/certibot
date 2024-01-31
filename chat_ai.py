def get_ai_response(user_input):
    # A simple rule-based AI for demonstration
    if "bonjour" in user_input.lower():
        return "Bonjour! Comment puis-je vous aider?"
    else:
        return "Je ne sais pas comment répondre à cela"
        