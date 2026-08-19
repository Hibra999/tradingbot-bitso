"""Authenticated dashboard service."""

from .app import DashboardController, EventHub, create_app

__all__ = ["DashboardController", "EventHub", "create_app"]
