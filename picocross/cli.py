import click


@click.command
@click.option("-n", "--name", help="Name to greet", type=str, required=True)
@click.option(
    "-m", "--message", help="Message to greet", type=str, default="How are you?"
)
def cli(name, message):
    """The picocross CLI program to run cross-sections"""
    print(f"Hello {name}")
    print(message)
