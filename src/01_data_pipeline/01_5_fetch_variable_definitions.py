import wrds
import pandas as pd
import os

def main():
    print("Starte Abfrage der Variablen-Definitionen...")
    
    input_path = "data/03_processed/thesis_base_dataset_ftdrop.csv"
    output_dir = "results/01_desc_analysis"
    output_file = f"{output_dir}/variable_definitions_codebook.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path, nrows=0)
        my_columns = df.columns.tolist()
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    # 1. lokales Lexikon für ausführliche Beschreibungen auf Basis des Compustat WRDS Financial Items Dictionary
    # und dem Refinitiv ESG Point in Time Bulk Service
    detailed_descriptions = {
'gvkey' : 'Global Company Key: The Global Company Key is a unique six-digit number assigned by Compustat to each company in the database. It is the primary key used to identify companies across all Compustat files.',
'indfmt' : 'Industry Format: This item indicates whether a company\'s financial statements are presented in an Industrial or Financial services format. It determines the structure and standard data items available for the company.',
'datafmt' : 'Data Format: This item describes the source and format of the data. For example, it indicates whether the data is from standard standardized financial statements, or adapted for specific presentation formats like historical or restated data.',
'consol' : 'Level of Consolidation - Company Annual Descriptor: This item indicates the level of consolidation of the financial statements, such as Consolidated (C) or Parent company only (P), specifying how subsidiaries are treated in the financial data.',
'popsrc' : 'Population Source: This code identifies the population source or geographic market of the company, distinguishing between International (I) for companies outside North America and Domestic (D) for North American companies.',
'acctstd' : 'Accounting Standard: This item indicates the specific accounting standards used by the company to prepare its financial statements, such as US GAAP, IFRS, or domestic standards specific to the country of incorporation.',
'bspr' : 'Balance Sheet Presentation: This code indicates the specific presentation format of the balance sheet, such as unclassified or classified, which reflects how assets and liabilities are ordered and grouped.',
'curcd' : 'ISO Currency Code: This item represents the three-letter International Organization for Standardization (ISO) currency code, indicating the currency in which the company\'s financial data is reported.',
'final' : 'Final Indicator Flag: This flag indicates whether the data for the given fiscal period is considered final or if it is still subject to further updates and potential revisions by Compustat.',
'fyear' : 'Data Year - Fiscal: This item represents the fiscal year to which the company\'s annual financial data applies, determined based on the company\'s specific fiscal year-end month.',
'fyr' : 'Fiscal Year-end Month: This item indicates the month representing the end of the company\'s fiscal year. It is expressed as a two-digit number (e.g., 12 for December).',
'ismod' : 'Income Statement Model Number: This item indicates the specific layout or format model used by Compustat to standardize the presentation of the company\'s income statement based on its industry.',
'pddur' : 'Period Duration: This item indicates the number of months covered by the financial data period, typically 12 months for annual data and 3 months for quarterly data.',
'scf' : 'Cash Flow Format: This code indicates the format used by the company to present its Statement of Cash Flows, which determines which specific data items are available for cash flow analysis.',
'src' : 'Source Document: This item identifies the primary source document (such as a 10-K, annual report, or prospectus) from which the financial data for the specific period was collected.',
'upd' : 'Update Code: This code indicates the update status of the data record, reflecting whether the data is preliminary, fully processed, or has been subsequently corrected or revised.',
'datadate' : 'Data Date: This item represents the specific calendar date as of which the financial data is reported, typically corresponding to the company\'s fiscal period-end date.',
'fdate' : 'Final Date: This item represents the date on which Compustat considers the financial data for the given period to be finalized and fully updated in the database.',
'pdate' : 'Preliminary Date: This item represents the date on which the preliminary financial data or press release information for the period was first entered into the database.',
'aco' : 'Current Assets - Other - Total: This item represents the total value of current assets that are not classified into major categories like cash, receivables, or inventories, reflecting other short-term resources.',
 'acox' : 'Current Assets - Other - Sundry: This item represents miscellaneous current assets that are not classified elsewhere under cash, receivables, inventories, or other primary current asset categories.',
'act' : 'Current Assets - Total: This item represents the sum of cash, temporary investments, receivables, inventories, and other assets that are expected to be realized in cash or sold or consumed within one year or during the normal operating cycle of the business.',
'am' : 'Amortization of Intangibles: This item represents the systematic write-down of the cost of intangible assets (such as goodwill, patents, trademarks, or franchises) over their useful lives.',
'ao' : 'Assets - Other: This item represents long-term assets or noncurrent assets that cannot be properly classified into specific accounts like property, plant, and equipment, or long-term investments.',
'aox' : 'Assets - Other - Sundry: This item represents miscellaneous or minor noncurrent assets that are not separately disclosed or categorized within other long-term assets.',
'ap' : 'Accounts Payable - Trade: This item represents open account obligations owed to creditors for goods and services received in the ordinary course of business.',
'apch' : 'Accounts Payable - Increase (Decrease): This item represents the net change in accounts payable from the beginning to the end of the specified fiscal period as reported in the statement of cash flows.',
'apo' : 'Accounts Payable - Other: This item represents obligations owed to non-trade creditors, such as miscellaneous payables, non-trade expenses, or short-term liabilities not classified as trade accounts payable.',
'at' : 'Assets - Total: This item represents the sum of current assets, net property, plant, and equipment, and other noncurrent assets. It reflects the total resources owned or controlled by the company.',
'autxr' : 'Appropriations to Untaxed Reserves: This item represents transfers from retained earnings or current income to special untaxed, tax-deferred, or statutory reserves, often specific to certain international accounting jurisdictions.',
'caps' : 'Capital Surplus/Share Premium Reserve: This item represents the amount received by a company from shareholders in excess of the par or stated value of the shares issued. It is often referred to as Paid-in Capital in Excess of Par.',
'capx' : 'Capital Expenditures: This item represents the funds used by a company to acquire, upgrade, and maintain physical assets such as property, industrial buildings, or equipment (often reported as additions to property, plant, and equipment).',
'ceq' : 'Common/Ordinary Equity - Total: This item represents the common shareholders\' equity or net worth of the company, which includes common stock, capital surplus, and retained earnings, less treasury stock.',
'ch' : 'Cash: This item represents cash on hand, demand deposits with banks, and other balances that are immediately available for use in operations.',
'che' : 'Cash and Short-Term Investments: This item represents the sum of cash, cash equivalents, and highly liquid short-term investment securities that can be readily converted into cash.',
'cheb' : 'Cash and Cash Equivalents at Beginning of Year: This item represents the balance of cash, demand deposits, and highly liquid short-term investments available at the start of the specified fiscal year.',
'chech' : 'Cash and Cash Equivalents - Increase/(Decrease): This item represents the net change (inflow or outflow) in the company\'s total cash and cash equivalents during the fiscal period, as shown on the statement of cash flows.',
'chee' : 'Cash and Cash Equivalents at End of Year: This item represents the balance of cash, demand deposits, and highly liquid short-term investments available at the close of the specified fiscal year.',
'cogs' : 'Cost of Goods Sold: This item represents the direct costs attributable to the production or purchase of the goods sold by a company, including material and labor costs.',
'cstk' : 'Common/Ordinary Stock (Capital): This item represents the total nominal, par, or stated value of the common or ordinary shares outstanding that represent ownership in the corporation.',
   'dc' : 'Deferred Charges: This item represents long-term prepaid expenses or expenditures that are capitalized and systematically amortized over a future period exceeding one year.',
'dd1' : 'Long-Term Debt Due in One Year: This item represents the portion of long-term debt, including bonds, mortgages, and long-term notes, that matures within the next fiscal year.',
'dfxa' : 'Depreciation of Tangible Fixed Assets: This item represents the systematic allocation of the cost of tangible, physical fixed assets (such as plant, property, and equipment) over their useful lives.',
'dlc' : 'Debt in Current Liabilities - Total: This item represents the total amount of short-term notes payable, commercial paper, bank overdrafts, and the current portion of long-term debt due within one year.',
'dltt' : 'Long-Term Debt - Total: This item represents obligations with a maturity date extending beyond one year from the balance sheet date, such as bonds, debentures, mortgages, and long-term bank loans.',
'do' : 'Discontinued Operations: This item represents the income, gain, or loss resulting from the operations and disposal of a component or segment of a business that has been discontinued or sold.',
'dp' : 'Depreciation and Amortization: This item represents the combined total non-cash expense of depreciation on tangible assets and amortization on intangible assets charged against income during the period.',
'dpact' : 'Depreciation, Depletion and Amortization (Accumulated): This item represents the total cumulative amount of depreciation, depletion, and amortization charged against the historical cost of the company\'s fixed assets.',
'dpc' : 'Depreciation and Amortization (Cash Flow): This item represents the add-back of non-cash depreciation and amortization expenses to net income under the operating activities section of the statement of cash flows.',
'dv' : 'Cash Dividends (Cash Flow): This item represents the total cash outflow for dividends paid to shareholders on common and preferred stock during the period, as reported in the statement of cash flows.',
'ea' : 'Exchange Adjustments (Assets): This item represents the adjustments made to asset values on the balance sheet resulting from the fluctuation and translation of foreign currencies.',
'ebit' : 'Earnings Before Interest and Taxes: This item represents a company\'s operating profitability before deducting interest expenses and corporate income taxes, derived as operating income plus non-operating revenue.',
'ebitda' : 'Earnings Before Interest: This item represents Earnings Before Interest, Taxes, Depreciation, and Amortization, providing an indicator of a company\'s core operational cash profitability.',
'emp' : 'Employees: This item represents the total number of people employed by the company as of the end of the fiscal year, or an average number of employees during the year, as disclosed.',
'ero' : 'Equity Reserves - Other: This item represents miscellaneous, non-distributable, or other statutory equity reserves that are not classified as retained earnings or standard capital surplus.',
'exre' : 'Exchange Rate Effect: This item represents the net effect of exchange rate fluctuations on cash and cash equivalents held in foreign currencies, as reported on the statement of cash flows.',
'fca' : 'Foreign Exchange Income (Loss): This item represents the net gains or losses resulting from foreign currency transactions or translation adjustments included in the determination of net income.',
'fiao' : 'Financing Activities - Other: This item represents miscellaneous cash inflows or outflows related to financing activities that are not separately categorized under debt or equity issuance/repayment.',
'fincf' : 'Financing Activities - Net Cash Flow: This item represents the net cash flow resulting from financing activities, including the net proceeds or payments from debt, stock, and dividends.',
'fopo' : 'Funds from Operations - Other: This item represents miscellaneous adjustments or non-cash items included in funds from operations that are not captured by major operating adjustment accounts.',
'gdwl' : 'Goodwill: This item represents the excess of the cost of an acquired entity over the net of the amounts assigned to assets acquired and liabilities assumed.',
'ib' : 'Income Before Extraordinary Items: This item represents the income of a company after all operating and non-operating revenues and expenses, income taxes, and minority interest have been deducted, but before extraordinary items and discontinued operations.',
'ibc' : 'Income Before Extraordinary Items (Cash Flow): This item represents income before extraordinary items as reported on or reconciled to the statement of cash flows, serving as the starting point for calculating operating cash flows under the indirect method.',
'icapt' : 'Invested Capital - Total: This item represents the sum of long-term debt, preferred stock, minority interest, and total common equity. It reflects the total long-term capital invested in the company.',
'idit' : 'Interest and Related Income - Total: This item represents interest income and other financial revenues earned by the company from cash balances, short-term securities, or loans extended to others.',
'intan' : 'Intangible Assets - Total: This item represents the total value of non-physical assets, such as patents, copyrights, trademarks, franchises, goodwill, and software, that provide long-term value to the company.',
'invch' : 'Inventory - Decrease (Increase): This item represents the net change in inventories from the beginning to the end of the period, as reflected in the operating activities section of the statement of cash flows.',
'invfg' : 'Inventories - Finished Goods: This item represents the cost of manufactured products that are ready for sale or distribution to customers.',
'invo' : 'Inventories - Other: This item represents miscellaneous inventories that cannot be categorized as raw materials, work in process, or finished goods (e.g., spare parts or operating supplies).',
'invrm' : 'Inventories - Raw Materials: This item represents the cost of basic commodities, components, and materials purchased by the company that have not yet undergone manufacturing or processing.',
'invt' : 'Inventories - Total: This item represents the total cost of merchandise held for sale, raw materials, work in process, and finished goods, valued at the lower of cost or market.',
'invwip' : 'Inventories - Work In Process: This item represents the cost of partially completed goods or products that are currently in the production pipeline and require further processing before sale.',
'ivaco' : 'Investing Activities - Other: This item represents miscellaneous cash inflows or outflows from investing activities that are not separately disclosed under capital expenditures or investments in securities.',
'ivaeq' : 'Investment and Advances - Equity: This item represents the long-term carrying value of investments in affiliated companies or joint ventures accounted for under the equity method.',
'ivao' : 'Investment and Advances - Other: This item represents long-term investments in non-affiliated companies, non-current marketable securities, real estate held for investment, or advances to partners.',
'ivncf' : 'Investing Activities - Net Cash Flow: This item represents the net cash flow resulting from all investing activities, including capital expenditures, acquisitions, and the purchase or sale of investment securities.',
'ivst' : 'Short-Term Investments - Total: This item represents temporary, highly liquid investments in marketable securities, commercial paper, or short-term bonds that are intended to be sold within one year.',
'lco' : 'Current Liabilities - Other - Total: This item represents the total of miscellaneous short-term obligations and current liabilities that are not classified into major accounts like accounts payable or short-term debt.',
'lcox' : 'Current Liabilities - Other - Sundry: This item represents minor or incidental current liabilities that are not separately categorized or itemized within other current liabilities.',
'lct' : 'Current Liabilities - Total: This item represents the sum of obligations that are expected to be satisfied within one year or during the normal operating cycle, including accounts payable, short-term debt, and accrued expenses.',
'lo' : 'Liabilities - Other - Total: This item represents the total of long-term or noncurrent liabilities that are not classified into major categories such as long-term debt or deferred taxes.',
'lse' : 'Liabilities and Stockholders Equity - Total: This item represents the sum of total liabilities, noncontrolling interests, and total stockholders\' equity, which equals total assets.',
'lt' : 'Liabilities - Total: This item represents the sum of current liabilities and all noncurrent/long-term liabilities owed by the company.',
'ltdch' : 'Long-Term Debt - Change: This item represents the net increase or decrease in long-term debt during the fiscal period, as reported on the statement of cash flows.',
'mibn' : 'Noncontrolling Interests - Nonredeemable - Balance Sheet: This item represents the equity interest in a subsidiary company that is owned by minority shareholders and cannot be redeemed or bought back by the parent company.',
'mibt' : 'Noncontrolling Interests - Total - Balance Sheet: This item represents the total carrying amount of equity in consolidated subsidiaries that is not attributable, directly or indirectly, to the parent company.',
'mii' : 'Noncontrolling Interest (Income Account): This item represents the portion of a consolidated subsidiary\'s net income or loss that belongs to minority or noncontrolling shareholders rather than the parent company.',
'nopi' : 'Nonoperating Income (Expense): This item represents revenues and expenses resulting from secondary or incidental activities of a company, rather than its core operations (e.g., interest income, dividend income, or gains/losses on asset sales).',
'np' : 'Notes Payable - Short-Term Borrowings: This item represents short-term obligations evidenced by formal promissory notes, including bank loans, commercial paper, and overdrafts.',
'oancf' : 'Operating Activities - Net Cash Flow: This item represents the net cash flow generated or used by a company\'s core business operations, calculated by adjusting net income for non-cash items and changes in working capital.',
'oiadp' : 'Operating Income After Depreciation: This item represents a company\'s profitability from core business operations after deducting operating expenses, including cost of goods sold, SG&A, and depreciation and amortization.',
'oibdp' : 'Operating Income Before Depreciation: This item represents a company\'s revenue minus operating expenses, excluding depreciation and amortization, often used as a measure of core operational profitability.',
'pi' : 'Pretax Income: This item represents a company\'s net income or loss before the deduction of corporate income tax expense.',
'ppegt' : 'Property, Plant and Equipment - Total (Gross): This item represents the total historical cost of physical, long-term assets utilized in operations (such as land, buildings, and machinery) before deducting accumulated depreciation.',
'ppent' : 'Property, Plant and Equipment - Total (Net): This item represents the gross value of property, plant, and equipment minus the total accumulated depreciation, reflecting the remaining book value of physical assets.',
'prc' : 'Participation Rights Certificates: This item represents special equity or non-voting financial instruments common in certain international jurisdictions (like Germany or Switzerland) that grant holders a share in profits and liquidation proceeds.',
'psfix' : 'Proceeds From Sale of Fixed Assets: This item represents the cash inflow received by the company from the sale or disposal of property, plant, and equipment, as reported under investing activities in the statement of cash flows.',
'pstk' : 'Preferred/Preference Stock (Capital) - Total: This item represents the total nominal, par, or stated value of preferred stock outstanding, which carries preferential rights over common stock regarding dividends and liquidation.',
'pstkn' : 'Preferred/Preference Stock - Nonredeemable: This item represents preferred stock that does not contain provisions for mandatory redemption or redemption at the option of the holder.',
'pstkr' : 'Preferred/Preference Stock - Redeemable: This item represents preferred stock that is subject to mandatory redemption by the company or is redeemable at the option of the shareholder at a specified date or price.',
're' : 'Retained Earnings: This item represents the cumulative net income or loss of a company from its inception, less any dividends distributed to shareholders and any transfers to other capital accounts.',
'recch' : 'Accounts Receivable - Decrease (Increase): This item represents the net change in accounts receivable from the beginning to the end of the period, as reported in the operating activities section of the statement of cash flows.',
'recco' : 'Receivables - Current - Other: This item represents short-term amounts owed to the company from sources other than regular trade customers, such as employees, affiliates, or tax authorities.',
'rect' : 'Receivables - Total: This item represents the sum of all short-term amounts owed to the company, including trade accounts receivable, notes receivable, and miscellaneous current receivables, net of allowances for doubtful accounts.',
'rectr' : 'Receivables - Trade: This item represents the amounts owed to the company by customers for goods sold or services rendered in the ordinary course of business, net of allowances for doubtful accounts.',
'revt' : 'Revenue - Total: This item represents the total gross operating revenue of a company, including sales, service revenues, and other inflows from core business operations, net of discounts, returns, and allowances.',
'rvlrv' : 'Revaluation Reserve: This item represents the equity reserve arising from the revaluation of assets (such as property, plant, and equipment) above their historical cost, typically used under non-US accounting standards like IFRS.',
'rvutx' : 'Reserves - Untaxed: This item represents untaxed or tax-deferred reserves set aside by companies under specific international accounting and tax laws, which have not yet been subject to corporate income tax.',
'sale' : 'Sales/Turnover (Net): This item represents the gross sales or turnover generated from the core operations of the company, less returns, allowances, and sales discounts.',
'sco' : 'Share Capital - Other: This item represents the value of miscellaneous or alternative forms of share capital that cannot be classified into standard common or preferred stock categories.',
'seq' : 'Stockholders Equity - Parent: This item represents the total equity interest attributable to the shareholders of the parent company, excluding noncontrolling/minority interests in subsidiaries.',
'spi' : 'Special Items: This item represents unusual or nonrecurring gains or losses that are included in operating income but are separated out because of their nature (e.g., restructuring charges, impairments, or write-downs).',
'sppiv' : 'Sale of Property, Plant and Equipment and Investments - Gain (Loss): This item represents the net gain or loss recognized by the company upon the sale or disposal of its fixed assets or investment securities.',
'teq' : 'Stockholders Equity - Total: This item represents the total equity of the entity, including common stock, capital surplus, retained earnings, treasury stock, and any noncontrolling/minority interests.',
'transa' : 'Cumulative Translation Adjustment: This item represents the cumulative component of other comprehensive income resulting from the translation of foreign subsidiary financial statements into the reporting currency of the parent company.',
'tsca' : 'Treasury Stock (Current Asset): This item represents the cost of a company\'s own shares that it has repurchased and holds temporarily as a short-term or current asset, intended for resale or imminent redistribution.',
'tstk' : 'Treasury Stock - Total (All Capital): This item represents the total cost or par value of a company\'s own shares that have been repurchased and are held by the issuing corporation, reducing total stockholders\' equity.',
'tstlta' : 'Treasury Stock (Long-Term Asset): This item represents the cost of a company\'s own repurchased shares that are classified as long-term investments or noncurrent assets under specific international accounting frameworks.',
'txc' : 'Income Taxes - Current: This item represents the estimated amount of income taxes payable to federal, state, and foreign governments based on the taxable income for the current fiscal period.',
'txdb' : 'Deferred Taxes (Balance Sheet): This item represents the net balance sheet asset or liability reflecting the future tax consequences of temporary differences between the financial reporting and tax bases of assets and liabilities.',
'txdi' : 'Income Taxes - Deferred: This item represents the non-cash tax expense or benefit resulting from timing differences between financial accounting and tax reporting, causing deferred tax assets or liabilities to change during the period.',
'txditc' : 'Deferred Taxes and Investment Tax Credit: This item represents the balance sheet accumulation of deferred income taxes resulting from temporary differences, combined with unamortized investment tax credits.',
'txo' : 'Income Taxes - Other: This item represents miscellaneous income tax expenses or credits that are not classified as current or deferred, such as adjustments for prior-year tax audits or special statutory levies.',
'txp' : 'Income Taxes Payable: This item represents the current liability for federal, state, local, or foreign taxes that are owed based on taxable income and must be paid within the next fiscal year.',
'txpd' : 'Income Taxes Paid: This item represents the actual amount of cash outflow for income taxes made to government authorities during the period, as reported in the statement of cash flows or supplemental disclosures.',
'txt' : 'Income Taxes - Total: This item represents the total burden of income taxes charged against current period earnings, comprising the sum of current tax expense and deferred tax expense or benefit.',
'txw' : 'Excise Taxes: This item represents taxes levied on the manufacture, sale, or consumption of specific commodities (such as fuel, tobacco, or alcohol) that are included in the company\'s expenses or revenue.',
'unl' : 'Unappropriated Net Loss: This item represents the accumulated net loss of a company that has not been allocated to specific reserves or offset by subsequent earnings within stockholders\' equity.',
'unnp' : 'Unappropriated Net Profit (Stockholders\'\' Equity): This item represents the portion of net profit or retained earnings that has not been allocated or transferred to specific reserves and remains available for dividend distribution.',
'wcap' : 'Working Capital (Balance Sheet): This item represents the excess of total current assets over total current liabilities, serving as a primary metric for assessing a company\'s short-term liquidity and operational health.',
'wcapopc' : 'Working Capital/Net Operating Assets - Change: This item represents the net change in working capital components or operating assets and liabilities from the beginning to the end of the period, as shown on the statement of cash flows.',
'xacc' : 'Accrued Expenses: This item represents liabilities for expenses that have been incurred but are not yet invoiced, billed, or paid as of the balance sheet date (e.g., accrued wages or interest).',
'xido' : 'Extraordinary Items and Discontinued Operations: This item represents the combined total of financial impacts resulting from discontinued operations and events classified as extraordinary due to being both unusual and infrequent.',
'xint' : 'Interest and Related Expense - Total: This item represents the total cost of borrowing funds during the period, including interest on short-term and long-term debt, amortization of debt discount, and related financial expenses.',
'xopr' : 'Operating Expenses - Total: This item represents the total expenses incurred from a company\'s primary operations, encompassing the cost of goods sold, selling, general and administrative expenses, and depreciation.',
'xopro' : 'Operating Expense - Other: This item represents miscellaneous operating expenses incurred in the normal course of business that are not categorized under cost of goods sold, SG&A, or standard operating lines.',
'xpp' : 'Prepaid Expenses: This item represents current assets arising from expenditures for benefits or services (such as insurance, rent, or utilities) that will be received or consumed in a future short-term period.',
'xrent' : 'Rental Expense: This item represents the total cost incurred by a company during the fiscal period for renting or leasing property, plant, equipment, or facilities under operating lease agreements.',
'xsga' : 'Selling, General and Administrative Expense: This item represents the sum of all direct and indirect selling expenses, along with all general and administrative expenses of the company (excluding cost of goods sold and interest).',
'iid' : 'Header Global Issue ID: This item represents the unique identifier assigned by Compustat to a specific security issue of a company, used as a primary key to track equity issues on an international level.',
'exchg' : 'Stock Exchange Code: This item indicates the specific stock exchange on which the company\'s security is primarily listed or traded, using Compustat\'s numeric exchange codes.',
'isin' : 'International Security Identification Number: This item represents the International Security Identification Number (ISIN), an international standard code that uniquely identifies a specific security issue.',
'sedol' : 'SEDOL: This item represents the Stock Exchange Daily Official List (SEDOL) code, which is a unique seven-character identifier assigned to securities traded on the London Stock Exchange and other international exchanges.',
'ajexi' : 'Adjustment Factor (International Issue)-Cumulative by Ex-Date: This item represents the cumulative factor used to adjust historical international stock price and share data for corporate actions like stock splits, stock dividends, and rights offerings.',
'curcdi' : 'ISO Currency Code - Security Financial Annual Descriptor: This item represents the three-letter ISO currency code indicating the currency in which the security-level financial data (such as stock prices or dividends) is reported.',
'cshoi' : 'Com Shares Outstanding - Issue: This item represents the total number of common or ordinary shares outstanding for a specific security issue as of the company\'s fiscal year-end.',
'cshpria' : 'Common Shares Used to Calculate Earnings Per Share (Basic) - As Reported: This item represents the weighted average number of common shares outstanding during the period used by the company to calculate basic earnings per share, as originally reported in its financial statements.',
'epsexcon' : 'Earnings Per Share (Basic) - Excluding Extraordinary Items - Consolidated: This item represents the basic earnings per share from consolidated operations, calculated by dividing income before extraordinary items attributable to common shareholders by the weighted average shares outstanding.',
'epsincon' : 'Earnings Per Share (Basic) - Including Extraordinary Items - Consolidated: This item represents the basic earnings per share from consolidated operations, calculated by dividing net income including extraordinary items attributable to common shareholders by the weighted average shares outstanding.',
'nicon' : 'Net Income (Loss) - Consolidated: This item represents the total consolidated net income or loss of the company, including the portions attributable to both the parent company shareholders and noncontrolling interests.',
'conm' : 'Company Name: This item represents the official legal name of the corporation as recorded in the Compustat database.',
'costat' : 'Active/Inactive Status Marker: This code indicates whether the company is currently active (A) or inactive/dead (I) in the database due to liquidation, merger, acquisition, or bankruptcy.',
'fic' : 'Current ISO Country Code - Incorporation: This item represents the three-letter ISO country code identifying the country in which the company is legally incorporated.',
'loc' : 'Current ISO Country Code - Headquarters: This item represents the three-letter ISO country code identifying the country where the company\'s main executive offices or corporate headquarters are located.',
'naicsh' : 'North America Industrial Classification System - Historical: This item represents the historical North American Industry Classification System (NAICS) code that was assigned to the company for the specific fiscal year, reflecting its primary business activity at that time.',
'sich' : 'Standard Industrial Classification - Historical: This item represents the historical Standard Industrial Classification (SIC) code that was assigned to the company for the specific fiscal year, reflecting its primary industry classification at that time.',
'rank' : 'Rank - Auditor: This code indicates the ranking or tier of the company\'s auditing firm (e.g., distinguishing between Big 4/Big 5 auditors and regional or smaller firms).',
'au' : 'Auditor: This item represents the specific numeric code assigned by Compustat to identify the independent public accounting firm that audited the company\'s financial statements.',
'auop' : 'Auditor Opinion: This code indicates the type of audit opinion issued by the independent auditor on the company\'s financial statements, such as unqualified, qualified, adverse, or a disclaimer of opinion.',
'hiid' : 'Historic Global Issue ID: This item represents the historical unique identifier assigned by Compustat to track a specific security issue during a specific historical period.',
'orgpermid' : 'Organization PermID: This unique identifier is assigned by Refinitiv to permanently link and track an organization across all Refinitiv platforms and content sets, ensuring consistent entity symbology.',
'year' : 'Data Year / Fiscal Year: This item represents the specific calendar or fiscal year to which the company\'s reported ESG data points, metrics, and calculated scores apply.',
'ticker' : 'Ticker Symbol: This item represents the stock ticker symbol assigned to the company\'s primary equity security on its major listing exchange, used for easy company identification.',
'comname' : 'Company Name: This item represents the official legal name of the organization or corporation as registered and standardized within the Refinitiv database.',
'fieldid' : 'Field ID: This item represents the unique numeric identifier assigned by Refinitiv to a specific ESG data point, category, or calculated analytic within the ESG framework.',
'hierarchy' : 'Hierarchy Level / Path: This item indicates the structural placement of the field within the Refinitiv ESG data hierarchy, defining how data points roll up into categories, pillars, and final scores.',
'fieldname' : 'Field Name: This item represents the official label or descriptive title of the ESG data item or metric as defined in the Refinitiv ESG Point in Time Glossary.',
'valuedate' : 'Value Date / Point-in-Time Date: This item represents the specific date on which the data point value was recorded, updated, or made publicly available in the Refinitiv database, crucial for point-in-time backtesting.',
'value' : 'Value: This item represents the raw reported data value or indicator response (which can be numeric, text, or a boolean boolean/flag) for the specific ESG metric prior to scoring.',
'valuescore' : 'Value Score / Analytic Value Score: This item represents the normalized, benchmarked, or percentile-ranked score calculated for an individual data point or indicator based on its raw value relative to its industry or country peer group.',
'esg_score' : 'ESG Score: This item represents the overall comprehensive ESG score, which is an unweighted average of the three pillar scores (Environmental, Social, and Governance), reflecting the company\'s relative ESG performance, commitment, and effectiveness.',
'env_score' : 'Environmental Pillar Score: This item represents the consolidated score for the Environmental pillar, measuring a company\'s impact on the living and non-living natural systems, encompassing resource use, emissions, and environmental product innovation.',
'gov_score' : 'Governance Pillar Score: This item represents the consolidated score for the Corporate Governance pillar, measuring a company\'s systems and processes to ensure its board members and executives act in the best interests of long-term shareholders, encompassing management, shareholders, and CSR strategy.',
'soc_score' : 'Social Pillar Score: This item represents the consolidated score for the Social pillar, measuring a company\'s capacity to generate trust and loyalty with its workforce, customers, and society, encompassing workforce, human rights, community, and product responsibility.',

    }

    # 2. WRDS Verbindung und Metadaten-Abfrage
    db = wrds.Connection()
    library = 'comp'
    table = 'g_funda'
    
    print(f"Rufe WRDS-Labels für {library}.{table} ab...")
    try:
        dict_df = db.describe_table(library, table)
        dict_df['name'] = dict_df['name'].str.lower()
        
        # Mapping für das kurze WRDS-Label (comment)
        if 'comment' in dict_df.columns:
            desc_map = dict(zip(dict_df['name'], dict_df['comment']))
        else:
            desc_map = {}
            
    except Exception as e:
        print(f"Fehler bei WRDS-Abfrage: {e}")
        desc_map = {}
    
    # 3. Eigene Spalten mit WRDS-Labels UND lokalen Beschreibungen matchen
    codebook_data = []
    for col in my_columns:
        col_lower = col.lower()
        
        # Kurzes Label von WRDS
        wrds_label = desc_map.get(col_lower, "Kein Label in WRDS gefunden")
        
        # Ausführliche Beschreibung aus lokalem Lexikon
        long_desc = detailed_descriptions.get(col_lower, "---")
        
        codebook_data.append({
            'Variable': col, 
            'WRDS_Label': wrds_label, 
            'Ausfuehrliche_Beschreibung': long_desc
        })
        
    codebook_df = pd.DataFrame(codebook_data)
    codebook_df.to_csv(output_file, index=False)
    
    print(f"Codebook mit erweiterten Beschreibungen generiert!")
    print(f"Gespeichert unter: {output_file}")

if __name__ == "__main__":
    main()