"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
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
    """Raised when DGL Graph is failed to create."""
    message_template = "Failure during creating DGLGraph."
    error_code: str = "DGL_CREATE_ERROR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InvalidFeatTypeError(BaseApplicationError):
    """Raised when the features field is missing."""
    message_template = ("The 'features' field in the JSON payload "
                        "request must be a dictionary.")
    error_code: str = "INVALID_FEATURE_TYPE"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MissingValError(BaseApplicationError):
    """Raised when a value is missing."""
    message_template = "Missing required {value_name} in {input_name}."
    error_code: str = "MISSING_VALUE_ERROR"

    def __init__(self, value_name, input_name, **kwargs):
        super().__init__(value_name=value_name,
                         input_name=input_name, **kwargs)


class MissingKeyError(BaseApplicationError):
    """Raised when a required key is missing."""
    message_template = "Missing required key: {key_name} in {input_name}."
    error_code: str = "MISSING_KEY_ERROR"

    def __init__(self, key_name, input_name, **kwargs):
        super().__init__(key_name=key_name, input_name=input_name, **kwargs)


class MisMatchedTypeError(BaseApplicationError):
    """Raised when a required node/edge type is missing
    in graph construction config."""
    message_template = ("Non-existed {structure_type} {type_name} defined "
                        "in graph construction config.")
    error_code = "MISMATCHED_COLUMN_ERROR"

    def __init__(self, structure_type, type_name, **kwargs):
        super().__init__(structure_type=structure_type, type_name=type_name, **kwargs)


class MisMatchedFeatureError(BaseApplicationError):
    """Raised when a required node/edge id has mismatched feature name."""
    message_template = ("Non-existed feature keys for {structural_type} {id_name}, "
                        "Expected Keys: {expected_keys}, Got Keys: {actual_keys}.")
    error_code = "MISMATCHED_FEATURE_ERROR"

    def __init__(self, structural_type, id_name, expected_keys, actual_keys, **kwargs):
        super().__init__(structural_type=structural_type, id_name=id_name,
                         expected_keys=expected_keys, actual_keys=actual_keys, **kwargs)
