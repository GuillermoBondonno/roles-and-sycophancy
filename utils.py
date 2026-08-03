import requests
import logging

logger = logging.getLogger(__name__)


def send_ntfy_message(message: str, topic: str = "roles-and-sycophancy-experiments") -> None:
    """
    Send a message to the ntfy.sh notification service.

    Args:
        message (str): The message to send.
        topic (str): The topic to send the message to. Defaults to "roles-and-sycophancy-experiments".
    """
    url = f"https://ntfy.sh/{topic}"
    headers = {"Title": "Roles and Sycophancy Experiments"}
    response = requests.post(url, data=message.encode("utf-8"), headers=headers)
    logger.info(f"Sent ntfy message to topic '{topic}': {response.status_code} - {response.text}")
    
