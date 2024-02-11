# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MovieInfoItem(scrapy.Item):
    title = scrapy.Field()
    genre = scrapy.Field()
    director = scrapy.Field()
    country = scrapy.Field()
    year = scrapy.Field()
    rating = scrapy.Field()


