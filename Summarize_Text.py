from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

def main():
    print('Hi From Agentic AI Learning')
    print('Processing....')

    information = """INCOME TAX DEPARTMENT 
    Government of India 
    F.No.: ITO/Ward-4(2)/2023-24/Notice/1892 
    PAN: ABCDE1234F 
    Name: Rajesh Kumar Sharma 
    Address: 42, Lajpat Nagar, New Delhi - 110024 
    Assessment Year: 2023-24 
    Date of Notice: 15-March-2024 
    Subject: Intimation under Section 143(1) of the Income Tax Act, 1961 —  
    Tax Demand Notice 
    Dear Mr. Rajesh Kumar Sharma, 
    This is to inform you that your Income Tax Return (ITR-2) for the  
    Assessment Year 2023-24 filed on 28-July-2023 has been processed  
    by the Centralized Processing Centre (CPC), Bengaluru. 
    Upon processing, certain discrepancies have been identified between  
    the information furnished in your return and the data available with  
    the Income Tax Department through Form 26AS, Annual Information  
    Statement (AIS), and Taxpayer Information Summary (TIS). 
    DISCREPANCIES IDENTIFIED: 
    1. INTEREST INCOME NOT DECLARED - Interest income of Rs. 1,20,000/- credited to your savings  
    and fixed deposit accounts as reported by HDFC Bank and SBI  
    in Form 26AS has not been declared in your ITR under the head  
    "Income from Other Sources." - Tax impact: Rs. 37,200/- (at applicable slab rate of 31.2%) 
    2. TDS MISMATCH 
    - You have claimed TDS credit of Rs. 18,500/- under Section 194A. - However, as per Form 26AS, TDS of only Rs. 10,000/- has been  
    deposited by the deductors against your PAN. - Excess TDS credit claimed: Rs. 8,500/- 
    3. DEDUCTION UNDER SECTION 80C — PARTIAL DISALLOWANCE - Deduction claimed in ITR: Rs. 1,50,000/- - Amount eligible as per supporting documents submitted: Rs. 95,000/- - Disallowed amount: Rs. 55,000/- - Tax impact: Rs. 17,160/- 
    4. FOREIGN REMITTANCE INCOME (Section 56) - As per AIS, a foreign remittance of Rs. 80,000/- was received  
    in your account during FY 2022-23 from a source in the UAE. - This has not been reflected in your ITR. - Tax impact: Rs. 24,960/- 
    COMPUTATION OF DEMAND: 
    Total Additional Income Assessed:      
    Tax on Additional Income:              
    Less: Excess TDS Claimed:              
    Interest under Section 234A:           
    Interest under Section 234B:          
    Rs. 3,55,000/- 
    Rs. 79,320/- 
    Rs. 8,500/- 
    Rs. 4,758/- 
     Rs. 3,172/- 
    ───────────────────────────────────────────────── 
    Total Outstanding Tax Demand:         
     Rs. 95,750/- 
    ───────────────────────────────────────────────── 
    RESPONSE REQUIRED: 
    You are hereby required to respond to this notice within 30 days  
    from the date of receipt through the e-filing portal  
    (incometax.gov.in). You may: 
    a) Accept the demand and make payment through the portal. 
    b) Dispute the demand by submitting supporting documents and  
    clarifications online. 
    Failure to respond within the stipulated period may result in  
    recovery proceedings under Section 220(2) of the Income Tax Act, 1961,  
    and may also attract penalty under Section 271(1)(c) for  
    concealment of income. 
    Please quote the Document Identification Number (DIN):  
    ITBA/CPC/2023-24/1892/10045672 in all future correspondence  
    regarding this notice. 
    Yours faithfully, 
    Sd/- 
    Income Tax Officer 
    Ward 4(2), New Delhi 
    Contact: ito.ward4.delhi@incometax.gov.in 
    Helpline: 1800-103-0025 """



    summary_template = """
    You are an expert document analyst. Carefully read the following text:
    
    {information}
    
    Provide the Following:
    
    1. Summary: A concise 3-4 Line Overview
    2. Key point: Most important information in bullet point
    3. Critical Figure/ Dates: Any important numbers, amounts, or deadlines
    4. Action Required:  What needs to be done, by whom, and by when
    5. Risk/Consequences: What happens if action is not taken (if mentioned)
    """



    summary_prompt_template = PromptTemplate(
        input_variables= ["information"],
        template= summary_template
    )


    llm = ChatNVIDIA(temperature= 0, model= "meta/llama-3.1-8b-instruct")

    chain = summary_prompt_template | llm


    response = chain.invoke(input={"information": information})


    print(response.content)



main()






