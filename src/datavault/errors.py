class DatavaultError(Exception):
    exit_code = 1


class RepositoryNotFound(DatavaultError):
    pass


class RepositoryExists(DatavaultError):
    pass


class InvalidReference(DatavaultError):
    pass


class UnsafePath(DatavaultError):
    pass


class ConflictError(DatavaultError):
    pass


class LockTimeout(DatavaultError):
    pass


class CorruptRepository(DatavaultError):
    pass


class ValidationFailed(DatavaultError):
    pass


class ExperimentError(DatavaultError):
    pass


class AIError(DatavaultError):
    pass


class ProviderUnavailable(AIError):
    pass


class RateLimited(AIError):
    pass


class ProviderTimeout(AIError):
    pass


class InvalidModelOutput(AIError):
    pass


class RequestTooLarge(AIError):
    pass


class ExternalDisabled(AIError):
    pass
