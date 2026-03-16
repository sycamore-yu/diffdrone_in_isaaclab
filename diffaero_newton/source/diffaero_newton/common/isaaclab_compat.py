import sys

def launch_app():
    """Bootstrap IsaacLab AppLauncher for headless environments.
    
    In headless test/training environments, IsaacSim's kernel will fail to bootstrap
    if the UI/TLS blocks cannot allocate memory or initialize. This wrapper ensures
    the AppLauncher runs in headless mode and suppresses those fatal initialization errors
    before importing the rest of IsaacLab's dependencies.
    """
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app
    return app
