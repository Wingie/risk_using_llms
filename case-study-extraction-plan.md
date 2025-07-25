# Case Study Extraction and Organization Plan

## Problem Analysis
The current chapters contain valuable, detailed case studies mixed with core conceptual content. This creates:
- Overly long chapters (some 5,000+ lines)
- Dense, hard-to-navigate content
- Valuable case studies buried in technical explanations
- Inconsistent chapter lengths and organization

## Solution: Case Study Separation Strategy

### **Phase 1: Extract Case Studies into Standalone Chapters**

Create dedicated case study chapters following this naming convention:
- `case-001-samsung-chatgpt-leak.md` - Samsung semiconductor ChatGPT data leak (2023)
- `case-002-healthcare-insurance-claims.md` - Healthcare insurer LLM agent claims processing attack
- `case-003-travel-booking-injection.md` - Travel company indirect prompt injection
- `case-004-financial-advisor-extraction.md` - Trading strategy extraction via training data
- `case-005-corporate-knowledge-segmentation.md` - Manufacturing company segmentation attack
- `case-006-healthcare-vector-probing.md` - Healthcare provider vector database exploitation
- etc.

### **Phase 2: Standardize Case Study Format**

Each case study chapter should follow this structure:

```markdown
# Case Study #XXX: [Title]
**Industry**: [Sector]  
**Attack Type**: [Classification]  
**Impact**: $[Amount] / [Description]  
**Date**: [When occurred]  
**Sources**: [Documentation]

## Executive Summary
[2-3 paragraph overview]

## Background
[Organization and system description]

## The Attack
### Initial Access
### Technique Progression  
### Exploitation Phase

## Technical Analysis
[Detailed technical breakdown with code examples where relevant]

## Business Impact
[Financial, operational, regulatory consequences]

## Lessons Learned
[Key takeaways and preventive measures]

## Related Cases
[Links to similar incidents]
```

### **Phase 3: Streamline Main Chapters**

After extracting case studies, main chapters become:
- **Focused on core concepts and frameworks**
- **Reference case studies** via links: "See Case Study #005 for a detailed example"
- **Target length**: 1,500-2,500 lines
- **Better organization** with consistent structure

### **Phase 4: Create Case Study Index**

Create `case-studies-index.md` organizing cases by:
- **Attack Type**: Prompt injection, data poisoning, etc.
- **Industry**: Healthcare, finance, travel, manufacturing
- **Impact Level**: Low, medium, high, critical
- **Technical Complexity**: Basic, intermediate, advanced

## Benefits of This Approach

1. **Preserves Valuable Content**: All detailed case studies retained with full technical analysis
2. **Improves Readability**: Main chapters become focused and navigable
3. **Creates Reference Library**: Case studies become reusable across chapters
4. **Better Organization**: Consistent structure and logical flow
5. **Modular Updates**: Individual case studies can be updated as new information emerges

## Implementation Priority

### **High Priority Chapters** (Extract case studies first):
- Chapter 4: Invisible Data Leaks (2,257 lines) - Contains Samsung case and multiple detailed examples
- Chapter 5: Business Logic (3,022 lines) - Healthcare insurance and other detailed cases
- Chapter 8: Temporal Attacks (4,749 lines) - Multiple complex attack scenarios
- Chapter 10: Supply Chain (5,613 lines) - Numerous real-world breach examples

### **Case Studies to Extract First**:
1. Samsung ChatGPT semiconductor leak (Ch4)
2. Healthcare insurance claims processing attack (Ch5) 
3. Corporate knowledge assistant segmentation attack (Ch4)
4. Financial advisor trading strategy extraction (Ch4)
5. Travel booking indirect prompt injection (Ch4)
6. Healthcare vector database probing (Ch4)

This approach maintains all the valuable detailed content while creating much better organization and readability.