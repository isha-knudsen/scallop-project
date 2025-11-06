# This will be inserted into the _fetch_csv_data function
# We need to check the year and use daily files for 2024+

        # Check if we need daily files (2024+) or monthly zone files (pre-2024)
        if start_date.year >= 2024:
            # Use daily files for 2024+
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y_%m_%d")
                filename = f"AIS_{date_str}.zip"
                url = f"{self.BASE_URL_CSV}/{current_date.year}/{filename}"
                
                cache_file = self.cache_dir / filename
                
                # Check cache
                if cache_file.exists():
                    logger.info(f"Loading from cache: {cache_file}")
                    df = self._read_zipped_csv(cache_file)
                else:
                    try:
                        logger.info(f"Downloading: {url}")
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                            if response.status == 200:
                                content = await response.read()
                                with open(cache_file, 'wb') as f:
                                    f.write(content)
                                df = self._read_zipped_csv(cache_file)
                            else:
                                logger.warning(f"Failed to download {url}: {response.status}")
                                current_date += timedelta(days=1)
                                continue
                    except Exception as e:
                        logger.error(f"Error downloading {url}: {str(e)}")
                        current_date += timedelta(days=1)
                        continue
                
                # Filter to region
                df_filtered = self._filter_to_region(df, bounds)
                all_data.append(df_filtered)
                
                current_date += timedelta(days=1)
        else:
            # Use monthly zone files for pre-2024
