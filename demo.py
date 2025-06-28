from pydtmdl import DTMProvider

provider = DTMProvider.get_provider_by_code("srtm30")
print(f"Provider Name: {provider.name()}")
