"""Typed failures at the trusted worker / policy boundary."""


class ResearchError(RuntimeError):
    """An experiment needs attention before safe continuation."""


class InfrastructureError(RuntimeError):
    """The harness, transport, renderer, or physics backend failed."""


class PolicyRequestError(ValueError):
    """A policy supplied an invalid public RobotAPI request."""
