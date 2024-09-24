# rhythm-of-risk

Code to reproduce results behind the publication [The Rhythm of Risk: Exploring Spatio-temporal Patterns of Urban Vulnerability with Ambulance Calls Data (2024)](https://journals.sagepub.com/doi/10.1177/23998083241272095?icid=int.sj-full-text.citing-articles.1) by [Mikhail Sirenko](https://scholar.google.nl/citations?user=ZzHyCt0AAAAJ&hl=en), [Tina Comes](https://www.tudelft.nl/tbm/resiliencelab/people/tina-comes) and [Alexander Verbraeck](https://www.tudelft.nl/staff/a.verbraeck/?cHash=79d864d800b2d588772fbe7e1778ff03).

## Data

This study relies on 3 datasets:

1. Open and anonymized ambulance calls from the [P2000](https://nl.wikipedia.org/wiki/P2000_(netwerk)) system that can be collected from one of the dedicated websites, e.g., [p2000-online.net](https://www.p2000-online.net/) or [Alarmeringen.nl](https://alarmeringen.nl/). Note that we use the data for autumn seasons of 2017, 2018, and 2019.

2. Open socio-demographic and built environment data on 500 by 500 m2 grid by Statistics Netherlands or [kaart met statistieken](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/kaart-van-500-meter-bij-500-meter-met-statistieken) in Dutch.

3. City, district and neighbourhood shapefiles or [wijk- en buurtkaart](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2023) by Statistics Netherlands.

## Methodology

The study is focused on three largest cities of the Netherlands: The Hague, Rotterdam, and Amsterdam.

We analyze ambulance calls on a _typical autumn weekday_. A typical weekday is an average over the days of three autumn seasons for years of ambulance call data from 2017, 2018, and 2019. Previous research has indicated that analyzing autumn can help focus on the city without many external factors: carnivals, tourists, etc. Additionally, on weekdays, all three cities have more calls and less inter-city mobility than on weekends, simplifying the pattern interpretation.

Finally, we aggregate the calls over six four-hour bins and allocate them to one of the 500 by 500 m2 grid cells in each city.

![Data preparation pipeline](report/figures/appendix/figs2.jpg)

Please look at the [supplementary material](https://journals.sagepub.com/doi/suppl/10.1177/23998083241272095/suppl_file/sj-pdf-1-epb-10.1177_23998083241272095.pdf) for more information about all the choices and reasoning behind them.

## Authors

_Mikhail Sirenko_
