import textwrap

def row_to_prompt(row, date_col, indicator_col, value_col, region_col=None):
    lines = []
    lines.append("You are an expert statistical reporter. Given the following structured data row, write a short (1-3 sentence) factual summary suitable for a national statistics bulletin. Use precise wording and avoid hallucination. If any value is missing, say 'data unavailable'.")
    lines.append("Row:")
    lines.append(f"- {date_col}: {row.get(date_col, 'N/A')}")
    lines.append(f"- {indicator_col}: {row.get(indicator_col, 'N/A')}")
    lines.append(f"- {value_col}: {row.get(value_col, 'N/A')}")
    if region_col:
        lines.append(f"- {region_col}: {row.get(region_col, 'N/A')}")
    lines.append("Write the summary below:")
    return "\n".join(lines)

def fallback_template(prompt):
    try:
        lines = prompt.splitlines()
        data = {}
        for l in lines:
            if l.strip().startswith("- "):
                try:
                    k, v = l.strip()[2:].split(":", 1)
                    data[k.strip().lower()] = v.strip()
                except:
                    pass
        date = data.get("date", "N/A")
        indicator = data.get("indicator", "the indicator")
        value = data.get("value", "N/A")
        region = data.get("region", None)
        if region and region.lower() not in ("n/a", "none", ""):
            return f"In {date}, {indicator} for {region} was {value}."
        else:
            return f"In {date}, {indicator} was {value}."
    except Exception:
        return "Data unavailable."
