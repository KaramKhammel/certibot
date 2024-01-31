def get_ai_response(user_input):
    # A simple rule-based AI for demonstration
    if "Bonjour" in user_input.lower():
        return "Bonjour! Comment puis-je vous aider?"
    else:
        return "Je ne sais pas comment répondre à cela"
        