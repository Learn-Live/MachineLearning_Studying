---


---

<h1 id="texshop-modify-syntax-colors">TexShop modify Syntax colors</h1>
<p>The command should be excuted in ‘<em><strong>terminal</strong></em>’ in Mac.</p>
<pre><code>defaults write TexShop commentred 0.1 
</code></pre>
<p>Refer to  :<br>
<a href="https://goudarzirandom.blogspot.com/2017/06/how-to-change-syntax-coloring-in-texshop.html">How to change syntax coloring in TeXShop</a></p>
<p>If you are a heavy user of TeXShop you probably already used the preferences to setup your preferred color schema for the editor. But have you noticed that you can’t change the syntax coloring from the GUI preferences? At least <em><strong>[ I haven’t found how to do that ].</strong></em> Here is how you can change RGB channel of the syntax coloring:</p>
<pre><code>defaults write TeXShop commandred 0.8  
defaults write TeXShop commandgreen 0.0  
defaults write TeXShop commandblue 1.0    
</code></pre>
<p>You can substitute command with any of the following keywords:</p>
<pre><code>background  
command  
comment  
foreground  
index  
insert_point  
marker
</code></pre>
<p>I hope this has been helpful! :-)</p>
<p>Refer to :<br>
<a href="https://tex.stackexchange.com/questions/394531/change-colour-scheme-in-texshop-so-that-it-matches-texmaker">Change colour scheme in TeXShop so that it matches Texmaker</a></p>
<!-- Written with [StackEdit](https://stackedit.io/). -->

