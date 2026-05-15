class CeroneError(Exception):
    """Base exception for Cerone client errors."""


class CeroneHTTPError(CeroneError):
    """Raised when the Cerone API returns an HTTP error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CeroneConfigurationError(CeroneError):
    """Raised when Cerone client configuration is incomplete."""


class CeroneApprovalRequiredError(CeroneError):
    """Raised when Cerone flags an action and caller policy requires escalation."""


class CeroneActionRejectedError(CeroneError):
    """Raised when Cerone rejects an action."""
