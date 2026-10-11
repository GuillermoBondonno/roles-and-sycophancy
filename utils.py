import requests
import logging

logger = logging.getLogger(__name__)


def send_ntfy_message(message: str, topic: str) -> None:
    """
    Send a message to the ntfy.sh notification service.

    Args:
        message (str): The message to send.
        topic (str): The explicitly configured destination topic.
    """
    url = f"https://ntfy.sh/{topic}"
    headers = {"Title": "Roles and Sycophancy Experiments"}
    response = requests.post(
        url, data=message.encode("utf-8"), headers=headers, timeout=10
    )
    response.raise_for_status()
    logger.info("Sent ntfy message (%s)", response.status_code)
