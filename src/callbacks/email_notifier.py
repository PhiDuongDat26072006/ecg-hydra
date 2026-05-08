import smtplib
from email.mime.text import MIMEText
from lightning import Callback, Trainer, LightningModule
from typing import Any, Dict, Optional
import os

class EmailNotifier(Callback):
    """Sends an email notification when training finishes or fails.

    :param smtp_server: The SMTP server address (e.g., smtp.gmail.com).
    :param smtp_port: The SMTP server port (e.g., 587).
    :param sender_email: The email address sending the notification.
    :param sender_password: The password or app password for the sender email.
    :param recipient_email: The email address receiving the notification.
    :param subject_prefix: A prefix for the email subject line.
    """

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str,
        subject_prefix: str = "[ECG Training]",
    ):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.subject_prefix = subject_prefix
        
        self.enabled = all([sender_email, sender_password, recipient_email])
        if not self.enabled:
            print("\n[EmailNotifier] Missing credentials. Email notifications are disabled.")

    def _send_email(self, subject: str, body: str):
        if not self.enabled:
            return

        msg = MIMEText(body)
        msg['Subject'] = f"{self.subject_prefix} {subject}"
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
        except Exception as e:
            print(f"\n[EmailNotifier] Failed to send email: {e}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called when the train ends."""
        metrics = trainer.callback_metrics
        
        # Format metrics into a readable string
        metrics_str = ""
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                metrics_str += f"- {k}: {v:.4f}\n"
            else:
                metrics_str += f"- {k}: {v}\n"
        
        best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "N/A")
        
        subject = "Training Finished Successfully"
        body = (
            f"Training session has completed.\n\n"
            f"Summary Metrics:\n{metrics_str}\n"
            f"Best Checkpoint: {best_model_path}\n\n"
            f"Check logs for more details."
        )
        self._send_email(subject, body)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException):
        """Called when any exception happens."""
        subject = "Training CRASHED"
        body = (
            f"The training process encountered an error and stopped.\n\n"
            f"Exception:\n{exception}\n\n"
            f"Please check the console/logs for the full traceback."
        )
        self._send_email(subject, body)
