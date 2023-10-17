# Configuration file for jupyter-server.

c = get_config()  # noqa

c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_root = True
c.ServerApp.open_browser = False

c.ServerApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$zmLak7HRgKR0PYRx4FEUnQ$mapmu0FltOr1lrcmlxenGBy04IoPF+nrL+Je4zCqM+A'
c.ServerApp.password_required = True

c.ServerApp.root_dir = '/home/ubuntu/multi-agent-service-selection'
c.ServerApp.default_url = '/tree'
