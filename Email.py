import smtplib
fromMy = 'b.rishabh@yahoo.co.in' # fun-fact: from is a keyword in python, you can't use it as variable, did abyone check if this code even works?
to  = 'rishabhbrajabasi@gmail.com'
subj='TheSubject'
date='2/1/2010'
message_text='Hello Or any thing you want to send'

msg = "From: %s\nTo: %s\nSubject: %s\nDate: %s\n\n%s" % ( fromMy, to, subj, date, message_text )

username = str('b.rishabh@yahoo.co.in')
password = str('Babama@1995')

server = smtplib.SMTP("smtp.mail.yahoo.com",587)
server.login(username,password)
server.sendmail(fromMy, to,msg)
server.quit()
print 'ok the email has sent '
