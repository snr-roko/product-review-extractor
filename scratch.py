import time

import ollama

from data import reviews_data as reviews_data_file

ZERO_SHOT_SYSTEM_MESSAGE = """
**System Message**

**Model Role:** Extract Review Information

**Task:** Extract relevant information from a product review with the output being the json object only

**Constraints:**

* Input: A review string containing text data
* Output: A JSON object with the following keys:
 + `product_name`: The name of the product (string)
 + `price`: The price of the product (float or None)
 + `currency`: The currency of the price (string or None)
 + `sentiment`: The sentiment of the review (one of: `delighted`, `satisfied`, `neutral`,
`disappointed`, `frustrated`, `angry`)
 + `repurchase_intent`: The repurchase intent of the reviewer (one of: `yes`, `no`, `maybe`,
`unclear`)
 + `intent_evidence`: The intent evidence of the reviewer (string or None)
 + `issues_mentioned`: A list of issues mentioned in the review (list of strings)

**Note:** The model should not attempt to parse or extract any additional information beyond
what is explicitly requested in the output keys.

**Input Format:** The input review string should be in plain text format.

**Output Format:** The output JSON object should be formatted as follows:
{
    "product_name": "",
    "price": null,
    "currency": null,
    "sentiment": "",
    "repurchase_intent": "",
    "intent_evidence": null,
    "issues_mentioned": []
}
The model should fill in the corresponding values for each key based on the input review string.
"""

def extract_review_data(review):
    start = time.perf_counter()
    response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {   "role": "system", "content": ZERO_SHOT_SYSTEM_MESSAGE},
                {    "role": "user", "content": review}
            ],
            options={
                "temperature": 0
            }
        )
    print(f"{time.perf_counter() - start:.1f}s, {response['eval_count']} tokens")
    

    return response["message"]["content"]

def main(num_reviews: int = 1):
    for i in range(num_reviews):

        review = reviews_data_file.reviews_data[i]["text"]
        print(f"Review {i + 1}: {review}")

        response = extract_review_data(review)
        print(f"Extracted Data from AI: {response}")

if __name__ == "__main__":
    main(num_reviews=3)