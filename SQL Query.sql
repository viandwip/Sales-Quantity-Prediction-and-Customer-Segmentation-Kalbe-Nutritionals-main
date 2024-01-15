select "Marital Status", round(avg(age), 2) avg_age
from customer c 
group by "Marital Status";

select gender, round(avg(age), 2) avg_age
from customer c 
group by gender;

select storename, sum(qty) total_qty
from store s 
join "Transaction" t on s.storeid = t.storeid
group by storename
order by total_qty desc;

select "Product Name", sum(totalamount) total_amount
from product p 
join "Transaction" t on p.productid = t.productid
group by "Product Name"
order by total_amount desc;