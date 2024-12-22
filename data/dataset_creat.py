import random
import csv
from datetime import datetime, timedelta

# Simple messages that don't need templating
simple_messages = {
    "greeting": {
        "inputs": [
            "hi", "hey", "hello", "yo", "hi there", "heya", "sup", "hiya", "howdy",
            "morning", "afternoon", "evening", "greetings", "hey there", "hi hi",
            "hello hello", "hey hey", "heyy", "hii", "hello!", "hi!"
        ],
        "responses": [
            "Hi!", "Hey!", "Hello!", "Hi there!", "Hey there!", "Hello there!",
            "Hi! How can I help?", "Hey! What's up?", "Hello! How may I help you?",
            "Hi there! Need any help?", "Hey! How can I assist you?",
            "Hello! What brings you here?"
        ]
    },
    "goodbye": {
        "inputs": [
            "bye", "goodbye", "bye bye", "see ya", "cya", "good night", "night",
            "take care", "ttyl", "talk to you later", "later", "catch you later",
            "peace", "peace out", "gotta go", "bye!", "byee", "bai", "ok bye"
        ],
        "responses": [
            "Bye!", "Goodbye!", "See you!", "Take care!", "Bye bye!",
            "Have a great day!", "Take care, bye!", "Goodbye, take care!",
            "See you later!", "Thanks, bye!", "Bye for now!",
            "Have a good one!"
        ]
    }
}

# Complex intents with templates
complex_intents = {
    "order_status": {
        "inputs": [
            "Where is my order #{}?",
            "Can you check status of order #{}",
            "I need update on order #{}",
            "Track order #{}",
            "Order #{} status please"
        ],
        "responses": [
            "Your order #{} is {}. Estimated delivery: {}.",
            "I've checked order #{}. Status: {}. Expected arrival: {}.",
            "Order #{} is currently {}. Should arrive {}.",
        ]
    },
    "product_inquiry": {
        "inputs": [
            "Do you have {} in stock?",
            "I'm looking for {}",
            "Tell me about {}",
            "Price of {}?",
            "Information about {}"
        ],
        "responses": [
            "Yes, {} is in stock. The price is {} and {}.",
            "We have {}. It costs {} and comes with {}.",
            "{} is available. Priced at {} with {}."
        ]
    },
    "technical_support": {
        "inputs": [
            "{} not working",
            "Issue with {}",
            "Help with {}",
            "How to fix {}",
            "Problem with {}"
        ],
        "responses": [
            "To fix {}, try: {}. If that doesn't work, {}.",
            "For {} issues: First {}, then {}.",
            "I can help with {}. Start by {}, followed by {}."
        ]
    },
    "payment_issue": {
        "inputs": [
            "Payment failed for {}",
            "Can't pay for {}",
            "Issue paying {}",
            "Payment error {}",
            "Problem paying {}"
        ],
        "responses": [
            "For the payment issue with {}, please {}. Then {}.",
            "To resolve payment for {}, try {}. If needed, {}.",
            "I'll help with your {} payment. First {}, then {}."
        ]
    }
}


def generate_dataset(num_samples):
    all_data = []
    start_date = datetime.now() - timedelta(days=30)
    order_numbers = [f"{random.randint(10000, 99999)}" for _ in range(100)]
    products = ["iPhone", "laptop", "headphones", "smartwatch", "tablet", "camera", "speaker", "monitor", "keyboard",
                "mouse"]

    with open('chatbot_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['id', 'intent', 'user_input', 'bot_response', 'timestamp', 'confidence_score', 'conversation_id'])

        current_conversation = 1000
        messages_in_conversation = 0

        for i in range(num_samples):
            if messages_in_conversation == 0 or messages_in_conversation >= random.randint(2, 5):
                current_conversation += 1
                messages_in_conversation = 0

            # Determine message type and intent
            is_simple = random.random() < 0.3

            if messages_in_conversation == 0:
                intent = "greeting"
            elif messages_in_conversation > 2 and random.random() < 0.3:
                intent = "goodbye"
            else:
                if is_simple and random.random() < 0.5:
                    intent = random.choice(["greeting", "goodbye"])
                else:
                    intent = random.choice(list(complex_intents.keys()))

            # Generate content
            if intent in simple_messages:
                user_input = random.choice(simple_messages[intent]["inputs"])
                response = random.choice(simple_messages[intent]["responses"])
            else:
                template = random.choice(complex_intents[intent]["inputs"])
                response_template = random.choice(complex_intents[intent]["responses"])

                if intent == "order_status":
                    order_num = random.choice(order_numbers)
                    status = random.choice(["in transit", "processing", "shipped", "out for delivery"])
                    delivery = random.choice(["tomorrow", "in 2 days", "next week", "by end of week"])
                    user_input = template.format(order_num)
                    response = response_template.format(order_num, status, delivery)

                elif intent == "product_inquiry":
                    product = random.choice(products)
                    price = f"${random.randint(99, 999)}"
                    feature = random.choice(["free shipping", "1-year warranty", "20% discount", "gift wrapping"])
                    user_input = template.format(product)
                    response = response_template.format(product, price, feature)

                elif intent == "technical_support":
                    issue = random.choice(["wifi", "bluetooth", "login", "app", "update"])
                    step1 = random.choice(["restart the device", "clear cache", "check settings", "update software"])
                    step2 = random.choice(["contact support", "try safe mode", "reset to defaults"])
                    user_input = template.format(issue)
                    response = response_template.format(issue, step1, step2)

                else:  # payment_issue
                    item = random.choice(["order", "subscription", "service", "purchase"])
                    step1 = random.choice(
                        ["check card details", "try another payment method", "verify billing address"])
                    step2 = random.choice(["contact your bank", "try again in 1 hour", "clear browser cache"])
                    user_input = template.format(item)
                    response = response_template.format(item, step1, step2)

            timestamp = start_date + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )

            confidence = (
                round(random.uniform(0.95, 0.99), 2) if intent in ["greeting", "goodbye"] and len(
                    user_input.split()) <= 2
                else round(random.uniform(0.90, 0.98), 2) if intent in ["greeting", "goodbye"]
                else round(random.uniform(0.75, 0.95), 2)
            )

            row = [i + 1, intent, user_input, response, timestamp.isoformat(), confidence, current_conversation]
            writer.writerow(row)
            all_data.append(row)

            messages_in_conversation += 1

    return all_data


# Generate dataset
dataset = generate_dataset(10000)

# Print sample
print("\nSample of generated dataset (first 5 rows):")
print("id,intent,user_input,bot_response,timestamp,confidence_score,conversation_id")
for row in dataset[:5]:
    print(",".join(str(x) for x in row))