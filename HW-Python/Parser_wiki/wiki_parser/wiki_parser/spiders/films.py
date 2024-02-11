from typing import Iterable
import scrapy
from scrapy.http import Request
from wiki_parser.items import MovieInfoItem
import time

def get_str_from_table(table):
    span_container = table.css('span')
    a_tags = span_container.css('a')
    if a_tags:
        text_list = a_tags.css('::text').getall()
        combined_text = ', '.join(set(text_list))
    else:
        combined_text = span_container.css('::text').get()
    return combined_text


class FilmsSpider(scrapy.Spider):
    name = "films"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    def start_requests(self):
        URL = "https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%A4%D0%B8%D0%BB%D1%8C%D0%BC%D1%8B_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83"
        yield scrapy.Request(url = URL, callback=self.parse_page)

    def parse_page(self, response):

        for selector in response.css('div#mw-pages li'):
            selector.css('::text').get()
            film_url = "https://ru.wikipedia.org/" + selector.css('a::attr(href)').get()
            yield response.follow(film_url, callback = self.parse_film)
             

        next_page = response.xpath('//a[text()="Следующая страница"]/@href').get()

        if next_page:
            next_url = "https://ru.wikipedia.org/" + next_page
            yield response.follow(next_url, callback = self.parse_page)

    def parse_film(self, response):

        item = MovieInfoItem()
        item['title'] = response.css('th::text').get()
        item['genre'] = get_str_from_table(response.css('tr:contains("Жанры"), tr:contains("Жанр")'))
        item['director'] = get_str_from_table(response.css('tr:contains("Режиссёры"), tr:contains("Режиссёр")'))
        item['country'] = get_str_from_table(response.css('tr:contains("Страна"), tr:contains("Страны")'))
        item['year'] = get_str_from_table(response.css('tr:contains("Год")'))
        item['rating'] = None

        raiting_link = response.css('tr:contains("IMDb") span a::attr(href)').get()

        if raiting_link:
            yield scrapy.Request(raiting_link, callback=self.parse_rating, meta={'item': item})
        else:
            yield item

    def parse_rating(self, response):
        item = response.meta['item']
        item['rating'] = response.css('div.sc-acdbf0f3-0 span.sc-bde20123-1::text').get()

        yield item

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

