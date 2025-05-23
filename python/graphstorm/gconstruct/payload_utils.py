class BaseApplicationError(Exception):
    """
    Base class for all custom application errors.
    Handles message templating and common detail storage.
    """
    # A placeholder template; derived classes should override this.
    message_template: str = "An unexpected assertion error occurred."
    error_code: str = "GENERIC_ERROR"

    def __init__(self, details: dict = None, **kwargs):
        """
        Initializes the base application error.

        Args:
            details (dict, optional): A dictionary to hold additional, unstructured details.
                                      These are appended to the formatted message.
            **kwargs: Keyword arguments that will be used to format the `message_template`.
                      These arguments are also merged into the `details` for full context.
        """
        self.format_args = kwargs
        self.details = details if details is not None else {}
        self.details.update(self.format_args)

        super().__init__(self._format_full_message())

    def _format_full_message(self) -> str:
        """
        Formats the message_template using the provided arguments,
        and appends any additional details.
        """
        primary_message = self.message_template.format(**self.format_args)

        # Append structured details if they exist
        if self.details:
            detail_strings = [f"{key}={value}" for key, value in self.details.items()]
            return f"{primary_message} ({', '.join(detail_strings)})"
        return primary_message

    def __str__(self):
        return self._format_full_message()

    def get_error_code(self) -> str:
        """Returns the error code associated with this exception."""
        return self.error_code


class DGLCreateError(BaseApplicationError):
    """Raised when a required parameter is missing."""
    message_template = "Failure during creating DGLGraph."
    error_code: str = "DGL_CREATE_ERROR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InvalidFeatTypeError(BaseApplicationError):
    """Raised when a required parameter is missing."""
    message_template = ("The 'features' field in the JSON payload "
                        "request must be a dictionary.")
    error_code: str = "INVALID_FEATURE_TYPE"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MissingValError(BaseApplicationError):
    """Raised when a required parameter is missing."""
    message_template = "Missing required {value_name} in {input_name}."
    error_code: str = "MISSING_VALUE_ERROR"

    def __init__(self, value_name, input_name, **kwargs):
        super().__init__(value_name=value_name,
                         input_name=input_name, **kwargs)


class MissingColumnError(BaseApplicationError):
    """Raised when a required parameter is missing."""
    message_template = "Missing required column: {column_name} in {input_name}."
    error_code: str = "MISSING_COLUMN_ERROR"

    def __init__(self, column_name, input_name, **kwargs):
        super().__init__(column_name=column_name, input_name=input_name, **kwargs)


class MisMatchedTypeError(BaseApplicationError):
    """Raised when a required node/edge type is missing
    in graph construction config."""
    message_template = ("Non-existed {structure_type} {type_name} defined "
                        "in graph construction config.")
    error_code = "MISMATCHED_COLUMN_ERROR"

    def __init__(self, structure_type, **kwargs):
        super().__init__(structure_type=structure_type, type_name=type_name, **kwargs)


class MisMatchedFeatureError(BaseApplicationError):
    message_template = ("Non-existed feature keys for {structural_type} {id_name} defined, "
                        "Expected Keys: {expected_keys}, Got Keys: {actual_keys}.")
    error_code = "MISMATCHED_FEATURE_ERROR"

    def __init__(self, structural_type, id_name, expected_keys, actual_keys, **kwargs):
        super().__init__(structural_type=structural_type, id_name=id_name,
                         expected_keys=expected_keys, actual_keys=actual_keys, **kwargs)
