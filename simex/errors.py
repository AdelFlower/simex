"""Typed failures at the trusted worker / policy boundary."""


class InfrastructureError(RuntimeError):
    """The harness, transport, renderer, or physics backend failed."""


class PolicyRequestError(ValueError):
    """A policy supplied an invalid public RobotAPI request."""
