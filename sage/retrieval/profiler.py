"""
Class to create daily and global preference summaries from history
"""
from collections import defaultdict
import click
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from sage.utils.llm_utils import GPTConfig


class UserProfiler:
    """
    This class handles the dynamic preference understanding.
    It creates daily user preference insights based on daily interactions.
    It further aggregates the daily insights into a global understanding of
    the user's preferences.
    """

    def __init__(self) -> None:

        self.daily_preferences = defaultdict(dict)
        self.global_profiles = defaultdict(dict)
        self.llm = GPTConfig().instantiate()

    def print_daily_summary(
        self, user_name: str, date: str, daily_queries: str
    ) -> None:
        """Beatifully print the daily summaries"""

        click.secho("-" * 20, fg="green")
        click.secho(f"[User Profiler] Daily summary for {user_name}\n", fg="green")
        click.secho("-" * 20, fg="green")
        click.secho("Daily interactions", fg="blue")

        for query in daily_queries:
            click.secho(f"{date}:: {query}", fg="blue")
        click.secho("-" * 20, fg="green")
        click.secho("LLM analysis of user preferences:", fg="magenta")
        click.secho(self.daily_preferences[user_name][date], fg="magenta")

    def print_global_profiles(self) -> None:
        """Beatifully print the global profiles"""
        click.secho("-" * 20, fg="green")
        click.secho("[User Profiler] Global profiles \n", fg="green")

        for user_name, profile in self.global_profiles.items():
            click.secho(f"User {user_name}: {profile}", fg="cyan")

    def update_daily_user_preferences(
        self, user_name: str, daily_queries: str, date: str
    ) -> None:
        """Extract daily user preferences based on daily interactions"""

        template = """Based on the following interactions, please summarize {user_name}'s preferences. The history content:\n
        {history}
        {user_name}'s preferences are:"""

        prompt = PromptTemplate(
            input_variables=["history", "user_name"], template=template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        inputs = {"history": daily_queries, "user_name": user_name}
        response = chain.predict(**inputs)
        self.daily_preferences[user_name][date] = response.strip().rstrip()
        self.print_daily_summary(user_name, date, daily_queries)

    def create_global_user_profile(self, user_name: str) -> None:
        """Summarize the daily user preferences into a single global profile"""

        template = (
            """The following are the user's preferences throughout multiple days."""
        )

        for date, summary in self.daily_preferences[user_name].items():
            # Avoid error from langchain
            summary = summary.replace("{", "").replace("}", "")
            template += f"\n At {date}, the user preferences are {summary}"

        template += "\n Please provide a highly concise and general summary of the user's preferences. The output format should be {{sports: list, favorite_teams: list, shows_genre: list, movie_genre: list, favorite_shows: list, favorite_movies: list, genre_to_avoid: list}}.:"

        prompt = PromptTemplate.from_template(template)

        chain = LLMChain(llm=self.llm, prompt=prompt)

        response = chain.predict()

        self.global_profiles[user_name] = response

    def update_global_user_profile(self, user_name: str, entries: list[str]) -> None:
        """
        Update the user global profile using new entries
        Note: This is not used at the momemt because we need to only update the
        profile with the most relevant/used memories and not any new memory
        """

        profile = self.global_profiles[user_name]

        profile = profile.replace("\n", "").replace("{", "[").replace("}", "]")

        template = f"""The following is the user profile : {profile}.\n Your job is to update the user profile based on the following new entries"""

        for entry in entries:
            # Avoid error from langchain
            template += f"\n {entry}"
        prompt = PromptTemplate.from_template(template)

        chain = LLMChain(llm=self.llm, prompt=prompt)

        response = chain.predict()

        self.global_profiles[user_name] = response
